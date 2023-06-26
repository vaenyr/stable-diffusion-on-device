#include "qnn_context.h"
#include "errors.h"
#include "utils.h"
#include "logging.h"

#include <map>
#include <string>
#include <vector>

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
inline T resolve_symbol(void* libHandle, const char* symName, bool required=true) {
    T ptr = reinterpret_cast<T>(dlsym(libHandle, symName));
    if (ptr == nullptr && required) {
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


std::map<QnnBackend, std::weak_ptr<QnnApi>> _loaded_backends;

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);


void _free_dl(void* hnd) {
    if (hnd) {
        dlclose(hnd);
    }
}

template <class T, class... Args>
void _generic_qnn_api_call(T&& f, const char* name, const char* func, const char* file, const char* line, Args&&... args) {
    auto status = f(std::forward<Args>(args)...);
    if (status != QNN_SUCCESS) {
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, format("QNN function \"{}\" returned error: {}", name, status), func, file, line);
    }
}


}


std::shared_ptr<QnnApi> QnnApi::get(QnnBackend backend) {
    auto&& itr = _loaded_backends.lower_bound(backend);
    if (itr->first == backend && !itr->second.expired())
        return itr->second.lock();


    QnnApi* raw_ptr;
    try {
        raw_ptr = new QnnApi(backend);
    } catch (std::bad_alloc const&) {
        throw libsd_exception(ErrorCode::FAILED_ALLOCATION, "Could not allocate QnnBackendLibrary", __func__, __FILE__, STR(__LINE__));
    }

    auto ret = std::shared_ptr<QnnApi>(raw_ptr);
    _loaded_backends[backend] = ret;
    return ret;
}


QnnApi::QnnApi(QnnBackend backend) : backend(backend) {
    if (static_cast<int>(backend) < 0 || static_cast<int>(backend) >= _num_backend_libs)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Backend argument out of bounds", __func__, __FILE__, STR(__LINE__));

    // core interface
    const char* _backend_lib_name = _backend_to_lib[static_cast<int>(backend)];
    dl = std::unique_ptr(dlopen(_backend_lib_name, RTLD_NOW | RTLD_LOCAL), _free_dl);
    if (!dl)
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, "Could not load backend library: " + std::string(_backend_lib_name), __func__, __FILE__, STR(__LINE__));

    {
        auto&& query_fn = resolve_symbol<QnnInterfaceGetProvidersFn_t>(dl, "QnnInterface_getProviders");

        QnnInterface_t** providers = nullptr;
        unsigned int num_providers = 0;

        auto status = query_fn((const QnnInterface_t***)&providers, &num_providers);
        if (status != QNN_SUCCESS || providers == nullptr || num_providers == 0)
            throw libsd_exception(ErrorCode::RUNTIME_ERROR, format("Could not query available interface providers: {}, {}, {}", status, providers, num_providers), __func__, __FILE__, STR(__LINE__));

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
            throw libsd_exception(ErrorCode::RUNTIME_ERROR, "Could not find a suitable interface provider", __func__, __FILE__, STR(__LINE__));
        }

        Qnn_LogHandle_t _log_hnd = nullptr;
        if (QNN_SUCCESS != interface.logInitialize(qnn_log_callback, QNN_LOG_LEVEL_MAX, &_log_hnd)) {
            info("Warning: could not initialize QNN logging");
        } else {
            log_hnd = std::unique_ptr(_log_hnd, interface.logFree);
        }
    }

    // system interface
    system_dl = std::unique_ptr(dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL), _free_dl);
    if (!system_dl) {
        info("Warning: could not found libQnnSystem.so, some functions might fail");
    } else {
        auto&& query_fn = resolveSymbol<QnnSystemInterfaceGetProvidersFn_t>(system_dl, "QnnSystemInterface_getProviders", false);
        if (!query_fn) {
            info("Warning: could not resolve QnnSystemInterface_getProviders symbol, some functions might fail");
        } else {
            QnnSystemInterface_t** providers = nullptr;
            uint32_t num_providers = 0;

            auto status = query_fn((const QnnSystemInterface_t**)&systemInterfaceProviders, &numProviders);
            if (status != QNN_SUCCESS || providers == nullptr || num_providers == 0) {
                info("Warning: could not query available system interface providers: {}, {}, {}, some functions might fail", status, providers, num_providers);
            } else {
                bool found = false;
                for (unsigned int i = 0; i < num_providers; i++) {
                    if (QNN_SYSTEM_API_VERSION_MAJOR ==  providers[i]->systemApiVersion.major &&
                        QNN_SYSTEM_API_VERSION_MINOR <=  providers[i]->systemApiVersion.minor) {
                        found = true;
                        system_interface = providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
                        break;
                    }
                }

                if (!found) {
                    info("Warning: could not find a suitable system interface provider, some functions might fail");
                } else {
                    QnnSystemContext_Handle_t _system_hnd = nullptr;
                    if (QNN_SUCCESS != system_interface.systemContextCreate(&_system_hnd)) {
                        info("Warning: could not create QNN system context! Some functions might fail")
                    } else {
                        system_hnd = std:unique_ptr(_system_hnd, system_interface.systemContextFree);
                    }
                }
            }
        }
    }
}

QnnApi::~QnnApi() {
    _loaded_backends.erase(backend);
}


qnn_hnd<Qnn_BackendHandle_t> QnnApi::create_backend(QnnBackend_Config_t** cfg) const {
    Qnn_BackendHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.backendCreate, "backendCreate", __func__, __FILE__, STR(__LINE__), log_hnd.get(), cfg, &ret);
    return qnn_hnd(ret, interface.backendFree);
}


qnn_hnd<Qnn_DeviceHandle_t> QnnApi::create_device(QnnDevice_Config_t** cfg) const {
    Qnn_DeviceHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.deviceCreate, "deviceCreate", __func__, __FILE__, STR(__LINE__), log_hnd.get(), cfg, &ret);
    return qnn_hnd(ret, interface.deviceFree);
}


qnn_hnd<Qnn_ContextHandle_t> QnnApi::create_context(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, QnnContext_Config_t** cfg) const {
    Qnn_ContextHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.contextCreate, "contextCreate", __func__, __FILE__, STR(__LINE__), backend, device, cfg, &ret);
    return qnn_hnd(ret, interface.contextFree);
}


qnn_hnd<Qnn_ContextHandle_t> QnnApi::create_context(std::vector<unsigned char> const& buffer, Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, QnnContext_Config_t** cfg) const {
    Qnn_ContextHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.contextCreateFromBinary, "contextCreateFromBinary", __func__, __FILE__, STR(__LINE__), backend, deviceHandle, cfg, buffer.data(), buffer.size(), &ret, nullptr);
    return qnn_hnd(ret, interface.contextFree);
}


void QnnApi::register_op_package(std::string const& package_path, std::string const& package_interface_provider) const {
    throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
}


QnnContext::QnnContext(QnnBackend backend, std::list<std::string> const& op_packages, bool burst) 
    : api(QnnApi::get(backend)) {

    _init_backend();
    _init_device();
    _init_context();
}

void QnnContext::_init_backend() {
    backend_hnd = api->create_backend(nullptr);
}

void QnnContext::_init_device() {
    if (api->get_backend_type() == QnnBackend::HTP) {
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
        auto&& dev_config_array = { &config_item_1, &config_item_2, nullptr };

        device_hnd = api->create_device(dev_config_array);
    } else {
        device_hnd = api->create_device(nullptr);
    }
}

void QnnContext::load_context(std::string const& context_blob) {
    std::vector<unsigned char> buffer;
    if (!read_file_content(context_blob, buffer))
        throw libsd_exception(ErrorCode::INVALID_ARGUMENT, format("Could not read content of the context blob: {}", context_blob), __func__, __FILE__, STR(__LINE__));

    auto&& context_hnd = api->create_context(buffer, backend_hnd.get(), device_hnd.get(), nullptr);

    auto&& bin_info = api->get_binary_info(buffer);
    for (auto&& e : bin_info) {
        QnnSystemContext_GraphInfo_t* graphs = nullptr;
        uint32_t num_graphs = 0;
        if (e.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
            graphs = e.contextBinaryInfoV1.graphs;
            num_graphs = e.contextBinaryInfoV1.numGraphs;
        } else if (e.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
            graphs = e.contextBinaryInfoV2.graphs;
            num_graphs = e.contextBinaryInfoV1.numGraphs;
        } else
            throw libsd_exception(ErrorCode::INVALID_ARGUMENT, format("Unexpected binary info version: {}", e.version), __func__, __FILE__, STR(__LINE__));

        for (uint32_t i=0; i<num_graphs; ++i) {
            auto&& graph = graphs[i];
            if (graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                _QnnGraph parsed = _QnnGraph
            } else
                throw libsd_exception(ErrorCode::INVALID_ARGUMENT, format("Unexpected graph info version: {}", graph.version), __func__, __FILE__, STR(__LINE__));
        }
    }
}