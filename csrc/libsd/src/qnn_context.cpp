#include "qnn_context.h"
#include "errors.h"
#include "utils.h"
#include "logging.h"

#include <map>

#include <dlfcn.h>

#include <QnnGraph.h>
#include <QnnDevice.h>
#include <QnnHtpGraph.h>
#include <QnnHtpDevice.h>
#include <QnnHtpPerfInfrastructure.h>


using namespace libsd;

namespace {

void qnn_log_callback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp) {
    LogLevel sd_level = LogLevel::NOTHING;
    switch (level) {
    case QNN_LOG_LEVEL_ERROR:
        sd_level = LogLevel::ERROR;
        break;
    case QNN_LOG_LEVEL_WARN:
    case QNN_LOG_LEVEL_INFO:
        sd_level = LogLevel::INFO;
        break;
    case QNN_LOG_LEVEL_DEBUG:
    case QNN_LOG_LEVEL_VERBOSE:
    case QNN_LOG_LEVEL_MAX:
        sd_level = LogLevel::DEBUG;
        break;
    }
    if (!is_enabled(sd_level))
        return;

    int rem = snprintf(nullptr, 0, fmt, argp);
    if (rem < 0)
        return debug("Could not handle a message from QNN! snprintf returned negative value: {}", rem);

    
    std::string buff{ rem+1 };
    rem = snprintf(buff.data(), buff.length(), fmt, argp);
    if (rem != buff.length() - 1)
        return debug("snprintf making our life harder...");

    message(timestamp, sd_level, buff);
 }


template <class T>
inline T resolve_symbol(void* libHandle, const char* symName) {
    T ptr = reinterpret_cast<T>(dlsym(libHandle, symName));
    if (ptr == nullptr) {
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, format("Unable to access symbol {}. dlerror(): {}", symName, dlerror()), __func__, __FILE__, STR(__LINE__));
    }
    return ptr;
}


const char* _backend_to_lib[] = {
    "libQnnCpu.so",
    "libQnnGpu.so",
    "libQnnDsp.so",
    "libQnnHtp.so",
    "libQnnHta.so"
};

constexpr size_t _num_backend_libs = sizeof(_backend_to_lib) / sizeof(decltype(_backend_to_lib[0]));


std::map<QnnBackend, std::weak_ptr<QnnBackendLibrary>> _loaded_backends;

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);


}


std::shared_ptr<QnnBackendLibrary> QnnBackendLibrary::get(QnnBackend backend) {
    auto&& itr = _loaded_backends.lower_bound(backend);
    if (itr->first == backend && !itr->second.expired())
        return itr->second.lock();


    QnnBackendLibrary* raw_ptr;
    try {
        raw_ptr = new QnnBackendLibrary(backend);
    } catch (std::bad_alloc const&) {
        throw libsd_exception(ErrorCode::FAILED_ALLOCATION, "Could not allocate QnnBackendLibrary", __func__, __FILE__, STR(__LINE__));
    }

    auto ret = std::shared_ptr<QnnBackendLibrary>(raw_ptr);
    _loaded_backends[backend] = ret;
    return ret;
}


QnnBackendLibrary::QnnBackendLibrary(QnnBackend backend) : backend(backend) {
    if (static_cast<int>(backend) < 0 || static_cast<int>(backend) >= _num_backend_libs)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Backend argument out of bounds", __func__, __FILE__, STR(__LINE__));

    const char* _backend_lib_name = _backend_to_lib[static_cast<int>(backend)];
    dl = dlopen(_backend_lib_name, RTLD_NOW | RTLD_LOCAL);
    if (!dl)
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, "Could not load backend library: " + std::string(_backend_lib_name), __func__, __FILE__, STR(__LINE__));

    auto getInterfaceProviders = resolve_symbol<QnnInterfaceGetProvidersFn_t>(dl, "QnnInterface_getProviders");

    QnnInterface_t** providers = nullptr;
    unsigned int num_providers = 0;

    auto status = getInterfaceProviders((const QnnInterface_t***)&providers, &num_providers);
    if (status != QNN_SUCCESS || interfaceProviders == nullptr || num_providers == 0) {
        dlclose(dl);
        dl = nullptr;
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, format("Could not query available interface providers: {}, {}, {}", status, providers, num_providers), __func__, __FILE__, STR(__LINE__));
    }

    bool found = false;
    for (unsigned int i = 0; i < num_providers; i++) {
        if (QNN_API_VERSION_MAJOR == providers[i]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= providers[i]->apiVersion.coreApiVersion.minor) {
            found = true;
            interface = interfaceProviders[i]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!found) {
        dlclose(dl);
        dl = nullptr;
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, "Could not find a suitable interface provider", __func__, __FILE__, STR(__LINE__));
    }

    if (QNN_SUCCESS != interface.logInitialize(qnn_log_callback, QNN_LOG_LEVEL_MAX, &log_hnd)) {
        info("Warning: could not initialize QNN logging");
    }
}

QnnBackendLibrary::~QnnBackendLibrary() {
    if (dl)
        dlclose(dl);

    _loaded_backends.erase(backend);
}


QnnContext::QnnContext(QnnBackend backend, std::list<std::string> const& op_packages, bool burst) 
    : backend_lib(QnnBackendLibrary::get(backend)) {

    _init_backend();
    _init_device();
}

void QnnContext::_init_backend() {
    backend_hnd = backend_lib->create_backend(&backend_cfg);
}

void QnnContext::_init_device() {
    const QnnDevice_Config_t** dev_config_array = { nullptr };

    if (backend_lib->get_backend() == QnnBackend::HTP) {
        QnnHtpDevice_CustomConfig_t dev_config_soc;
        dev_config_soc.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        dev_config_soc.socModel = QNN_SOC_MODEL_SM8550;

        QnnHtpDevice_CustomConfig_t dev_config_arch;
        dev_config_arch.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
        dev_config_arch.arch.arch = QNN_HTP_DEVICE_ARCH_V73;
        dev_config_arch.arch.deviceId = 0;

        QnnDevice_Config_t config_item_1;
        config_item_1.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        config_item_1.customConfig = &dev_config_soc;
        QnnDevice_Config_t config_item_2;
        config_item_2.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        config_item_2.customConfig = &dev_config_arch;
        dev_config_array = { &config_item_1, &config_item_2, nullptr };
    }

    device_hdn = backend_lib->create_device(dev_config_array);
}

void QnnContext::_init_context() {
    
}