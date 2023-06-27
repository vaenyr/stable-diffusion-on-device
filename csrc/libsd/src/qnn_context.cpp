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
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>


using namespace libsd;

namespace {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);


const char* _backend_to_lib[] = {
    "libQnnCpu.so",
    "libQnnGpu.so",
    "libQnnDsp.so",
    "libQnnHtp.so",
    "libQnnHta.so"
};

constexpr size_t _num_backend_libs = sizeof(_backend_to_lib) / sizeof(decltype(_backend_to_lib[0]));

std::map<QnnBackendType, std::weak_ptr<QnnApi>> _loaded_backends;


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

    va_list argp_copy;
    va_copy(argp_copy, argp);

    int rem = std::vsnprintf(nullptr, 0, fmt, argp_copy);
    if (rem < 0)
        return debug("Could not handle a message from QNN! snprintf returned negative value: {}", rem);

    std::string buff(rem+1, '\0');
    rem = std::vsnprintf(&buff[0], buff.size(), fmt, argp);
    if (rem != buff.size()-1)
        return debug("getting printf to work as expected, so difficult... :(");

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


void _free_dl(void* hnd) {
    if (hnd) {
        dlclose(hnd);
    }
}

template <class T, class... Args>
void _generic_qnn_api_call(T&& f, const char* name, const char* func, const char* file, const char* line, Args&&... args) {
    debug("Calling QNN function: {}", name);
    auto status = f(std::forward<Args>(args)...);
    if (status != QNN_SUCCESS) {
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, format("QNN function \"{}\" returned error: {}", name, status), func, file, line);
    }
}

} // anonymous


std::shared_ptr<QnnApi> QnnApi::get(QnnBackendType backend) {
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


QnnApi::QnnApi(QnnBackendType backend) : backend(backend) {
    if (static_cast<int>(backend) < 0 || static_cast<int>(backend) >= _num_backend_libs)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Backend argument out of bounds", __func__, __FILE__, STR(__LINE__));

    // core interface
    const char* _backend_lib_name = _backend_to_lib[static_cast<int>(backend)];
    dl = std::shared_ptr<void>(dlopen(_backend_lib_name, RTLD_NOW | RTLD_LOCAL), _free_dl);
    if (!dl)
        throw libsd_exception(ErrorCode::RUNTIME_ERROR, "Could not load backend library: " + std::string(_backend_lib_name), __func__, __FILE__, STR(__LINE__));

    {
        auto&& query_fn = resolve_symbol<QnnInterfaceGetProvidersFn_t>(dl.get(), "QnnInterface_getProviders");

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
                interface = providers[i]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!found) {
            throw libsd_exception(ErrorCode::RUNTIME_ERROR, "Could not find a suitable interface provider", __func__, __FILE__, STR(__LINE__));
        }

        Qnn_LogHandle_t _log_hnd = nullptr;
        if (QNN_SUCCESS != interface.logCreate(qnn_log_callback, QNN_LOG_LEVEL_DEBUG, &_log_hnd)) {
            info("Warning: could not initialize QNN logging");
        } else {
            log_hnd = qnn_hnd<Qnn_LogHandle_t>(_log_hnd, interface.logFree);
        }
    }

    // system interface
    system_dl = std::shared_ptr<void>(dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL), _free_dl);
    if (!system_dl) {
        info("Warning: could not found libQnnSystem.so, some functions might fail");
    } else {
        auto&& query_fn = resolve_symbol<QnnSystemInterfaceGetProvidersFn_t>(system_dl.get(), "QnnSystemInterface_getProviders", false);
        if (!query_fn) {
            info("Warning: could not resolve QnnSystemInterface_getProviders symbol, some functions might fail");
        } else {
            QnnSystemInterface_t** providers = nullptr;
            uint32_t num_providers = 0;

            auto status = query_fn((const QnnSystemInterface_t***)&providers, &num_providers);
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
                        info("Warning: could not create QNN system context! Some functions might fail");
                    } else {
                        system_hnd = qnn_hnd<QnnSystemContext_Handle_t>(_system_hnd, system_interface.systemContextFree);
                    }
                }
            }
        }
    }

#ifdef __ANDROID__
    cdsp_dl = std::shared_ptr<void>(dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL), _free_dl);
    if (!cdsp_dl) {
        info("Warning: could not load libcdsprpc.so, zero-copy data transfer will be disabled!");
    } else {
        rpcmem_alloc = resolve_symbol<decltype(rpcmem_alloc)>(cdsp_dl.get(), "rpcmem_alloc", false);
        rpcmem_free = resolve_symbol<decltype(rpcmem_free)>(cdsp_dl.get(), "rpcmem_free", false);
        rpcmem_to_fd = resolve_symbol<decltype(rpcmem_to_fd)>(cdsp_dl.get(), "rpcmem_to_fd", false);
        if (!rpcmem_alloc || !rpcmem_free || !rpcmem_to_fd) {
            info("Warning: could not resolve all RPC symbols, zero-cost transfer will be disabled");
            cdsp_dl.reset();
        }
    }
#endif
}


QnnApi::~QnnApi() {
    _loaded_backends.erase(backend);
}


qnn_hnd<Qnn_BackendHandle_t> QnnApi::create_backend(const QnnBackend_Config_t** cfg) const {
    Qnn_BackendHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.backendCreate, "backendCreate", __func__, __FILE__, STR(__LINE__), log_hnd.get(), cfg, &ret);
    return qnn_hnd<Qnn_BackendHandle_t>(ret, interface.backendFree);
}


qnn_hnd<Qnn_DeviceHandle_t> QnnApi::create_device(const QnnDevice_Config_t** cfg) const {
    Qnn_DeviceHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.deviceCreate, "deviceCreate", __func__, __FILE__, STR(__LINE__), log_hnd.get(), cfg, &ret);
    return qnn_hnd<Qnn_DeviceHandle_t>(ret, interface.deviceFree);
}


qnn_hnd<Qnn_ContextHandle_t> QnnApi::create_context(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** cfg) const {
    Qnn_ContextHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.contextCreate, "contextCreate", __func__, __FILE__, STR(__LINE__), backend, device, cfg, &ret);
    return qnn_hnd<Qnn_ContextHandle_t>(ret, [this](Qnn_ContextHandle_t hnd) { interface.contextFree(hnd, nullptr); });
}


qnn_hnd<Qnn_ContextHandle_t> QnnApi::create_context(std::vector<unsigned char> const& buffer, Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** cfg) const {
    Qnn_ContextHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.contextCreateFromBinary, "contextCreateFromBinary", __func__, __FILE__, STR(__LINE__), backend, device, cfg, buffer.data(), buffer.size(), &ret, nullptr);
    return qnn_hnd<Qnn_ContextHandle_t>(ret, [this](Qnn_ContextHandle_t hnd) { interface.contextFree(hnd, nullptr); });
}


void QnnApi::register_op_package(std::string const& package_path, std::string const& package_interface_provider) const {
    throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
}


QnnSystemContext_BinaryInfo_t const& QnnApi::get_binary_info(std::vector<unsigned char>& buffer) const {
    if (!system_hnd)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Attempted to get binary info of a serialized context but system context has not been created - see previous warnings", __func__, __FILE__, STR(__LINE__));

    const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
    Qnn_ContextBinarySize_t binary_info_size = 0;
    _generic_qnn_api_call(system_interface.systemContextGetBinaryInfo, "systemContextGetBinaryInfo", __func__, __FILE__, STR(__LINE__), system_hnd.get(), buffer.data(), buffer.size(), &binary_info, &binary_info_size);
    if (!binary_info)
        throw libsd_exception(ErrorCode::INVALID_ARGUMENT, "Returned binary info is a nullptr!", __func__, __FILE__, STR(__LINE__));

    return *binary_info;
}


Qnn_GraphHandle_t QnnApi::retrieve_graph(Qnn_ContextHandle_t context, const char* graph_name) const {
    Qnn_GraphHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.graphRetrieve, "graphRetrieve", __func__, __FILE__, STR(__LINE__), context, graph_name, &ret);
    return ret; // graph handles do not need to be freed, so no need to wrap them in shared_ptr
}


std::pair<std::shared_ptr<void>,int> QnnApi::allocate_ion(uint32_t size) {
    if (!cdsp_dl)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Tried to allocate RPC memory without ION support", __func__, __FILE__, STR(__LINE__));

    auto&& ptr = std::shared_ptr<void>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size), rpcmem_free);
    if (!ptr)
        throw libsd_exception(ErrorCode::FAILED_ALLOCATION, "Failed to allocate RPC memory!", __func__, __FILE__, STR(__LINE__));

    int fd = rpcmem_to_fd(ptr.get());
    return std::make_pair(std::move(ptr), fd);
}


qnn_hnd<Qnn_MemHandle_t> QnnApi::mem_register(Qnn_ContextHandle_t ctx, Qnn_MemDescriptor_t desc) {
    Qnn_MemHandle_t ret = nullptr;
    _generic_qnn_api_call(interface.memRegister, "memRegister", __func__, __FILE__, STR(__LINE__), ctx, &desc, 1, &ret);
    return qnn_hnd<Qnn_MemHandle_t>(ret, [this](Qnn_MemHandle_t ptr){ interface.memDeRegister(&ptr, 1); });
}


void QnnTensor::activate() const {
    if (!batch_size)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Cannot activate QnnTensor with batch_size==0!", __func__, __FILE__, STR(__LINE__));

    if (is_ion) {
        target.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        target.v1.memHandle = data_hnd.get();
    } else {
        Qnn_ClientBuffer_t wrapper = QNN_CLIENT_BUFFER_INIT;
        wrapper.data = data.get();
        wrapper.dataSize = data_size;

        target.v1.memType = QNN_TENSORMEMTYPE_RAW;
        target.v1.clientBuf = wrapper;
    }
}


uint32_t QnnTensor::get_num_elements(Qnn_Tensor_t const& t, unsigned int batch_size) {
    return 0;
}


uint8_t QnnTensor::get_element_size(Qnn_Tensor_t const& t) {
    return 0;
}


QnnTensor::QnnTensor(QnnApi& api, Qnn_ContextHandle_t ctx,  Qnn_Tensor_t& desc, unsigned int batch_size) : batch_size(batch_size), target(desc) {
    if (!batch_size)
        return;

    data_size = get_num_elements(batch_size) * get_element_size();
    if (api.has_ion()) {
        std::tie(data, data_fd) = api.allocate_ion(data_size);

        Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
        desc.memShape = { target.v1.rank, target.v1.dimensions, nullptr };
        desc.dataType = desc.dataType;
        desc.memType = QNN_MEM_TYPE_ION;
        desc.ionInfo.fd = data_fd;
        data_hnd = api.mem_register(ctx, desc);
        is_ion = true;
    } else {
        data = std::shared_ptr<void>(new uint8_t[data_size], std::default_delete<uint8_t[]>());
        is_ion = false;
    }
}


QnnTensor::QnnTensor(QnnTensor const& other, Qnn_Tensor_t& target) : is_ion(other.is_ion), batch_size(other.batch_size), data(other.data), data_size(other.data_size),
    data_fd(other.data_fd), data_hnd(other.data_hnd), target(target) {
}


QnnTensor QnnGraph::allocate_input(unsigned int idx, unsigned batch) {
    if (idx >= inputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Input index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    auto&& _ctx = ctx.lock();
    if (!_ctx)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Trying to allocate memory while context has already been deleted!", __func__, __FILE__, STR(__LINE__));
    return QnnTensor(*api, _ctx.get(), inputs[idx], batch);
}


QnnTensor QnnGraph::attach_input(unsigned int idx, QnnTensor const& t) {
    if (idx >= inputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Input index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    return QnnTensor(t, inputs[idx]);
}


QnnTensor QnnGraph::allocate_output(unsigned int idx, unsigned batch) {
    if (idx >= outputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Output index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    auto&& _ctx = ctx.lock();
    if (!_ctx)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Trying to allocate memory while context has already been deleted!", __func__, __FILE__, STR(__LINE__));
    return QnnTensor(*api, _ctx.get(), outputs[idx], batch);
}


QnnTensor QnnGraph::attach_output(unsigned int idx, QnnTensor const& t) {
    if (idx >= outputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Outputs index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    return QnnTensor(t, outputs[idx]);
}


QnnGraph::QnnGraph(qnn_hnd<Qnn_ContextHandle_t> ctx, std::shared_ptr<QnnApi> api, const char* name, Qnn_Tensor_t* inputs, unsigned int num_inputs,
    Qnn_Tensor_t* outputs, unsigned int num_outputs, Qnn_GraphHandle_t graph)
    : name(name), inputs(std::span(inputs, num_inputs)), outputs(std::span(outputs, num_outputs)), graph(graph), ctx(ctx), api(api) {
}


QnnContext::QnnContext(qnn_hnd<Qnn_ContextHandle_t> ctx, std::vector<QnnGraph>&& graphs) : ctx(ctx), graphs(std::move(graphs)) {
}


QnnBackend::QnnBackend(QnnBackendType backend, std::list<std::string> const& op_packages, bool burst) 
    : api(QnnApi::get(backend)) {

    _init_backend();
    _init_device();
}


void QnnBackend::_init_backend() {
    backend_hnd = api->create_backend(nullptr);
}


void QnnBackend::_init_device() {
    if (api->get_backend_type() == QnnBackendType::HTP) {
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

        const QnnDevice_Config_t* dev_config_array[] = { &config_item_1, &config_item_2, nullptr };
        device_hnd = api->create_device(dev_config_array);
    } else {
        device_hnd = api->create_device(nullptr);
    }
}


graph_refs QnnBackend::load_context(std::string const& context_blob) {
    std::vector<unsigned char> buffer;
    if (!read_file_content(context_blob, buffer))
        throw libsd_exception(ErrorCode::INVALID_ARGUMENT, format("Could not read content of the context blob: {}", context_blob), __func__, __FILE__, STR(__LINE__));

    debug("Read {} bytes from file: {}", buffer.size(), context_blob);

    auto&& context_hnd = api->create_context(buffer, backend_hnd.get(), device_hnd.get(), nullptr);
    debug("Context handler created");

    std::vector<QnnGraph> graphs;
    graph_refs ret;

    debug("Investigating context binary info...");
    auto&& bin_info = api->get_binary_info(buffer);

    QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
    uint32_t num_graphs = 0;
    if (bin_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        graphs_info = bin_info.contextBinaryInfoV1.graphs;
        num_graphs = bin_info.contextBinaryInfoV1.numGraphs;
    } else if (bin_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        graphs_info = bin_info.contextBinaryInfoV2.graphs;
        num_graphs = bin_info.contextBinaryInfoV1.numGraphs;
    } else
        throw libsd_exception(ErrorCode::INVALID_ARGUMENT, format("Unexpected binary info version: {}", bin_info.version), __func__, __FILE__, STR(__LINE__));

    debug("{} graphs reported", num_graphs);
    for (uint32_t i=0; i<num_graphs; ++i) {
        auto&& graph_info = graphs_info[i];
        if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            auto&& graph_hnd = api->retrieve_graph(context_hnd.get(), graph_info.graphInfoV1.graphName);
            graphs.emplace_back(QnnGraph(context_hnd, api, graph_info.graphInfoV1.graphName, graph_info.graphInfoV1.graphInputs, graph_info.graphInfoV1.numGraphInputs, graph_info.graphInfoV1.graphOutputs, graph_info.graphInfoV1.numGraphOutputs, graph_hnd));
            ret.push_back(graphs.back());
        } else
            throw libsd_exception(ErrorCode::INVALID_ARGUMENT, format("Unexpected graph info version: {}", graph_info.version), __func__, __FILE__, STR(__LINE__));
    }

    ctx.emplace_back(QnnContext(std::move(context_hnd), std::move(graphs)));
    return ret;
}


graph_refs QnnBackend::load_model(std::string const& model_so) {
    throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
}


// tensor_refs QnnBackend::run(graph_ref graph, tensor_refs& inputs) {
//     throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
// }
