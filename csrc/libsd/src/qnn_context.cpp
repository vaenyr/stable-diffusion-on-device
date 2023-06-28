#include "qnn_context.h"
#include "errors.h"
#include "utils.h"
#include "logging.h"

#include <map>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

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
        sd_level = LogLevel::INFO;
        break;
    case QNN_LOG_LEVEL_INFO:
    case QNN_LOG_LEVEL_VERBOSE:
        sd_level = LogLevel::DEBUG;
    case QNN_LOG_LEVEL_DEBUG:
    case QNN_LOG_LEVEL_MAX:
        sd_level = LogLevel::ABUSIVE;
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

std::string _format_to_str(Qnn_TensorDataFormat_t tformat) {
    if (tformat == 0)
        return "flat_buffer";
    return format("unk({})", hex(tformat));
}

std::string _dtype_to_str(Qnn_DataType_t dtype) {
    static const char* _names[] = {
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "unk(0x0208)", "float16", "float32", "unk(0x0264)",
        "sq8", "sq16", "sq32", "unk(0x0364)",
        "uq8", "uq16", "uq32", "unk(0x0464)",
        "bool"
    };

    uint8_t group = (dtype >> 8);
    if (group > 5 || dtype > 0x0508)
        return format("unk({})", hex(dtype));

    uint8_t bits = (dtype & 0xFF);
    if (bits != 0x08 && bits != 0x16 && bits != 0x32 && bits != 0x64)
        return format("unk({})", hex(dtype));

    bits = (bits >> 8) + (bits >> 1 & 0x01) + (bits >> 1 & 0x02);
    assert(bits >= 0 && bits <= 3);
    assert(group <= 5);
    assert(group < 5 || bits == 0);
    return _names[group*4 + bits];
}

std::string _ttype_to_str(Qnn_TensorType_t ttype) {
    switch (ttype) {
    case QNN_TENSOR_TYPE_APP_WRITE: return "w";
    case QNN_TENSOR_TYPE_APP_READ: return "r";
    case QNN_TENSOR_TYPE_APP_READWRITE: return "rw";
    case QNN_TENSOR_TYPE_NATIVE: return "h";
    case QNN_TENSOR_TYPE_STATIC: return "w";
    case QNN_TENSOR_TYPE_NULL: return "?";
    default:
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Unexpected tensor type: {}", hex(ttype)), __func__, __FILE__, STR(__LINE__));
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


QnnDevice_Infrastructure_t QnnApi::get_device_infrastructure() const {
    QnnDevice_Infrastructure_t ret = nullptr;
    _generic_qnn_api_call(interface.deviceGetInfrastructure, "deviceGetInfrastructure", __func__, __FILE__, STR(__LINE__), &ret);
    return ret;
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


void QnnApi::set_graph_config(Qnn_GraphHandle_t graph, const QnnGraph_Config_t** cfg) const {
    _generic_qnn_api_call(interface.graphSetConfig, "graphSetConfig", __func__, __FILE__, STR(__LINE__), graph, cfg);
}


std::pair<std::shared_ptr<void>,int> QnnApi::allocate_ion(uint32_t size) {
    if (!cdsp_dl)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Tried to allocate RPC memory without ION support", __func__, __FILE__, STR(__LINE__));

    auto&& ptr = std::shared_ptr<void>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size), [this](void* ptr){
        debug("Freeing RPC memory: {}", ptr);
        rpcmem_free(ptr);
    });
    debug("RPC memory allocated: {}, {}", ptr.get(), size);
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


void QnnApi::execute_graph(Qnn_GraphHandle_t graph, std::span<Qnn_Tensor_t> const& inputs, std::span<Qnn_Tensor_t>& outputs) {
    _generic_qnn_api_call(interface.graphExecute, "graphExecute", __func__, __FILE__, STR(__LINE__), graph, inputs.data(), inputs.size(), outputs.data(), outputs.size(), nullptr, nullptr);
}


QnnTensor::QnnTensor(QnnTensor&& other) : is_ion(other.is_ion), batch_size(other.batch_size), data(std::move(other.data)), data_size(other.data_size), data_fd(other.data_fd), data_hnd(std::move(other.data_hnd)), slot(other.slot) {
    if (slot.current_tensor == &other)
        slot.current_tensor = this;
}


QnnTensor::~QnnTensor() {
    if (data.get()) {
        debug("Deallocating a tensor pointing to the memory location: {}", data.get());
        deactivate();
    }
}


uint32_t QnnTensor::get_num_elements(Qnn_Tensor_t const& t, unsigned int batch_size) {
    if (!t.v1.rank)
        return 0;
    uint32_t ret = 1;
    for (uint32_t i=1; i<t.v1.rank; ++i)
        ret *= t.v1.dimensions[i];
    return ret*batch_size;
}


uint8_t QnnTensor::get_element_size(Qnn_Tensor_t const& t) {
    switch (t.v1.dataType & 0xFF) {
    case 0x08: return 1;
    case 0x16: return 2;
    case 0x32: return 4;
    case 0x64: return 8;
    default:
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Unexpected tensor data type! {}, lower 8-bit: {}", hex(t.v1.dataType), hex(t.v1.dataType & 0xFF)), __func__, __FILE__, STR(__LINE__));
    }
}


bool QnnTensor::is_quantized(Qnn_Tensor_t const& t) {
    return (t.v1.dataType >> 8) == 0x03 || (t.v1.dataType >> 8) == 0x04;
}


bool QnnTensor::is_floating_point(Qnn_Tensor_t const& t) {
    return (t.v1.dataType >> 8) == 0x03 || (t.v1.dataType >> 8) == 0x04 || (t.v1.dataType >> 8) == 0x02; // quantized or normal fp
}


void QnnTensor::activate() const {
    if (!batch_size)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Cannot activate QnnTensor with batch_size==0!", __func__, __FILE__, STR(__LINE__));

    if (slot.current_tensor == this)
        return;

    if (is_ion) {
        slot.target.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        slot.target.v1.memHandle = data_hnd.get();
    } else {
        Qnn_ClientBuffer_t wrapper = QNN_CLIENT_BUFFER_INIT;
        wrapper.data = data.get();
        wrapper.dataSize = data_size;

        slot.target.v1.memType = QNN_TENSORMEMTYPE_RAW;
        slot.target.v1.clientBuf = wrapper;
    }

    slot.current_tensor = this;
    debug("Memory location {} is now the source of data for slot: {}", data.get(), slot.target.v1.name);
}


void QnnTensor::deactivate() const {
    if (!batch_size)
        return;

    if (slot.current_tensor != this)
        return;

    Qnn_ClientBuffer_t wrapper = QNN_CLIENT_BUFFER_INIT;
    wrapper.data = nullptr;
    wrapper.dataSize = 0;

    slot.target.v1.memType = QNN_TENSORMEMTYPE_RAW;
    slot.target.v1.clientBuf = wrapper;

    slot.current_tensor = nullptr;
    debug("Slot {} is now unbounded, previous memory location: {}", slot.target.v1.name, data.get());
}


QnnTensor::QnnTensor(QnnApi& api, Qnn_ContextHandle_t ctx, graph_slot& slot, unsigned int batch_size) : batch_size(batch_size), slot(slot) {
    if (!batch_size)
        return;

    data_size = get_num_elements(batch_size) * get_element_size();
    if (api.has_ion()) {
        std::tie(data, data_fd) = api.allocate_ion(data_size);

        Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
        desc.memShape = { slot.target.v1.rank, slot.target.v1.dimensions, nullptr };
        desc.dataType = slot.target.v1.dataType;
        desc.memType = QNN_MEM_TYPE_ION;
        desc.ionInfo.fd = data_fd;
        data_hnd = api.mem_register(ctx, desc);
        is_ion = true;
        debug("New ION tensor allocated: {}; target: {}, {}, {}", data.get(), slot.target.v1.name, _dtype_to_str(slot.target.v1.dataType), std::span(slot.target.v1.dimensions, slot.target.v1.rank));
    } else {
        data = std::shared_ptr<void>(new uint8_t[data_size], [](void* ptr) {
            debug("Freeing memory: {}", ptr);
        });
        debug("Memory allocated: {}, {}", data.get(), data_size);
        is_ion = false;
        debug("New standard tensor allocated: {}; target: {}, {}, {}", data.get(), slot.target.v1.name, _dtype_to_str(slot.target.v1.dataType), std::span(slot.target.v1.dimensions, slot.target.v1.rank));
    }
}


QnnTensor::QnnTensor(QnnTensor const& other, graph_slot& slot) : is_ion(other.is_ion), batch_size(other.batch_size), data(other.data), data_size(other.data_size),
    data_fd(other.data_fd), data_hnd(other.data_hnd), slot(slot) {
    debug("New aliased tensor, data location: {} also targets {}, original target: {}", data.get(), slot.target.v1.name, other.slot.target.v1.name);
}


void QnnTensor::set_data(std::vector<float> const& buffer) {
    throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
}


void QnnTensor::set_data(std::vector<uint32_t> const& buffer) {
    throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
}


void QnnTensor::get_data(std::vector<float>& buffer) {
    throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
}


void QnnTensor::get_data(std::vector<uint32_t>& buffer) {
    throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Not implemented", __func__, __FILE__, STR(__LINE__));
}


QnnTensor QnnGraph::allocate_input(unsigned int idx, unsigned batch, bool activate) {
    if (idx >= inputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Input index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    auto&& _ctx = ctx.lock();
    if (!_ctx)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Trying to allocate memory while context has already been deleted!", __func__, __FILE__, STR(__LINE__));
    QnnTensor ret{ *api, _ctx.get(), input_slots[idx], batch };
    if (activate)
        ret.activate();
    return ret;
}


QnnTensor QnnGraph::attach_input(unsigned int idx, QnnTensor const& t, bool activate) {
    if (idx >= inputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Input index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    QnnTensor ret{ t, input_slots[idx] };
    if (activate)
        ret.activate();
    return ret;
}


QnnTensor QnnGraph::allocate_output(unsigned int idx, unsigned batch, bool activate) {
    if (idx >= outputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Output index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    auto&& _ctx = ctx.lock();
    if (!_ctx)
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, "Trying to allocate memory while context has already been deleted!", __func__, __FILE__, STR(__LINE__));
    QnnTensor ret{ *api, _ctx.get(), output_slots[idx], batch };
    if (activate)
        ret.activate();
    return ret;
}


QnnTensor QnnGraph::attach_output(unsigned int idx, QnnTensor const& t, bool activate) {
    if (idx >= outputs.size())
        throw libsd_exception(ErrorCode::INTERNAL_ERROR, format("Outputs index too large: {}", idx), __func__, __FILE__, STR(__LINE__));
    QnnTensor ret{ t, output_slots[idx] };
    if (activate)
        ret.activate();
    return ret;
}


void QnnGraph::verify() {
    std::list<std::string> missing;
    std::map<unsigned int, std::list<const char*>> batch_sizes;

    for (auto&& s : input_slots) {
        if (!s.current_tensor)
            missing.push_back(s.target.v1.name);
        else
            batch_sizes[s.current_tensor->batch_size].push_back(s.target.v1.name);
    }
    for (auto&& s : output_slots) {
        if (!s.current_tensor)
            missing.push_back(s.target.v1.name);
        else
            batch_sizes[s.current_tensor->batch_size].push_back(s.target.v1.name);
    }

    if (!missing.empty() || batch_sizes.size() > 1 || batch_sizes.empty()) {
        std::string batch_info = "";
        if (batch_sizes.empty()) {
            batch_info = "<no batch information>";
        }
        else if (batch_sizes.size() > 1) {
            for (auto&& i : batch_sizes)
                batch_info += format("\n    {}: {}", i.first, i.second);
        }
        throw libsd_exception(ErrorCode::RUNTIME_ERROR,
            format("Graph verification failed! At least one input or output tensor has not been assigned memory location and/or operates on different batch size!\n    Missing allocations: {}\n    Conflicting batch size: {}\n", missing, std::move(batch_info)),
            __func__, __FILE__, STR(__LINE__));
    }
}


void QnnGraph::execute() {
    api->execute_graph(graph, inputs, outputs);
}


QnnGraph::QnnGraph(qnn_hnd<Qnn_ContextHandle_t> ctx, std::shared_ptr<QnnApi> api, const char* name, Qnn_Tensor_t* inputs, unsigned int num_inputs,
    Qnn_Tensor_t* outputs, unsigned int num_outputs, Qnn_GraphHandle_t graph)
    : name(name), inputs(std::span(inputs, num_inputs)), outputs(std::span(outputs, num_outputs)), graph(graph), ctx(ctx), api(api) {
    if (is_enabled(LogLevel::DEBUG)) {
        debug("New graph: {}", name);
        debug("    Num inputs: {}", num_inputs);
        for (auto&& t : this->inputs) {
            debug("        {}: {}, {}, {}, {}", t.v1.name, _format_to_str(t.v1.dataFormat), _dtype_to_str(t.v1.dataType), _ttype_to_str(t.v1.type), std::span(t.v1.dimensions, t.v1.rank));
        }
        debug("    Num outputs: {}", num_outputs);
        for (auto&& t : this->outputs) {
            debug("        {}: {}, {}, {}, {}", t.v1.name, _format_to_str(t.v1.dataFormat), _dtype_to_str(t.v1.dataType), _ttype_to_str(t.v1.type), std::span(t.v1.dimensions, t.v1.rank));
        }
    }

    for (auto&& i : range(num_inputs))
        input_slots.emplace_back(graph_slot{ .target=inputs[i], .current_tensor=nullptr });
    for (auto&& i : range(num_outputs))
        output_slots.emplace_back(graph_slot{ .target=outputs[i], .current_tensor=nullptr });

    if (api->get_backend_type() == QnnBackendType::HTP) {
        // QnnHtpGraph_CustomConfig_t vtcm;
        // vtcm.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        // vtcm.vtcmSizeInMB = 8;

        // QnnHtpGraph_CustomConfig_t hvx;
        // hvx.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
        // hvx.numHvxThreads = 4;

        // QnnGraph_Config_t opt1;
        // opt1.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        // opt1.customConfig = &hvx;

        // QnnGraph_Config_t opt2;
        // opt1.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        // opt1.customConfig = &vtcm;
        // const QnnGraph_Config_t* options[] = { &opt1, &opt2, nullptr };

        // api->set_graph_config(graph, options);
    }
}


QnnContext::QnnContext(qnn_hnd<Qnn_ContextHandle_t> ctx, std::vector<QnnGraph>&& graphs) : ctx(ctx), graphs(std::move(graphs)) {
}


QnnBackend::QnnBackend(QnnBackendType backend, std::list<std::string> const& op_packages, bool burst) 
    : api(QnnApi::get(backend)), burst(burst) {

    _init_backend();
    _init_device();
    _init_performance();
}


QnnBackend::~QnnBackend() {
    if (_htp_power_config_id)
        _htp_perf_infra->destroyPowerConfigId(_htp_power_config_id.value());
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


void QnnBackend::_init_performance() {
    if (api->get_backend_type() != QnnBackendType::HTP || !burst)
        return;

    debug("Creating HTP power configurations");
    QnnDevice_Infrastructure_t deviceInfra = api->get_device_infrastructure();
    QnnHtpDevice_Infrastructure_t* htpInfra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(deviceInfra);
    _htp_perf_infra = htpInfra->perfInfra;

    _htp_power_config_id.emplace(1);
    uint32_t deviceId = 0;
    uint32_t coreId = 0;
    _htp_perf_infra->createPowerConfigId(deviceId, coreId, &_htp_power_config_id.value());

    //Initialize the power config and select the voltage corner values for the performance setting.
    _htp_burst_power_config.emplace();
    {
        auto&& power_config = _htp_burst_power_config.value();
        std::memset(&power_config, 0, sizeof(power_config));

        power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        power_config.dcvsV3Config.dcvsEnable = 1;
        power_config.dcvsV3Config.setDcvsEnable = 1;
        power_config.dcvsV3Config.contextId = _htp_power_config_id.value();

        // refer QnnHtpPerfInfrastructure.h
        power_config.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        power_config.dcvsV3Config.setSleepLatency = 1;//True to consider Latency parameter otherwise False
        power_config.dcvsV3Config.setBusParams = 1;//True to consider Bus parameter otherwise False
        power_config.dcvsV3Config.setCoreParams = 1;//True to consider Core parameter otherwise False
        power_config.dcvsV3Config.setSleepDisable = 0;//True to consider sleep disable/enable parameter otherwise False
        power_config.dcvsV3Config.sleepDisable = 0;//True to disable sleep, False to re-enable sleep

        //Set Sleep latency parameter
        power_config.dcvsV3Config.sleepLatency =  40;

        //set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
        power_config.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
        power_config.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
        power_config.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;

        //set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
        power_config.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
        power_config.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
        power_config.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    }

    _htp_normal_power_config.emplace();
    {
        auto&& power_config = _htp_normal_power_config.value();
        std::memset(&power_config, 0, sizeof(power_config));

        power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        power_config.dcvsV3Config.dcvsEnable = 1;
        power_config.dcvsV3Config.setDcvsEnable = 1;
        power_config.dcvsV3Config.contextId = _htp_power_config_id.value();

        // refer QnnHtpPerfInfrastructure.h
        power_config.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        power_config.dcvsV3Config.setSleepLatency = 1;//True to consider Latency parameter otherwise False
        power_config.dcvsV3Config.setBusParams = 1;//True to consider Bus parameter otherwise False
        power_config.dcvsV3Config.setCoreParams = 1;//True to consider Core parameter otherwise False
        power_config.dcvsV3Config.setSleepDisable = 0;//True to consider sleep disable/enable parameter otherwise False
        power_config.dcvsV3Config.sleepDisable = 0;//True to disable sleep, False to re-enable sleep

        //Set Sleep latency parameter
        power_config.dcvsV3Config.sleepLatency =  1000;

        //set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
        power_config.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
        power_config.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
        power_config.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;

        //set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
        power_config.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
        power_config.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
        power_config.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
    }
}


void QnnBackend::start_burst() {
    if (!burst || !_htp_power_config_id)
        return;

    debug("Switching to burst power mode...");
    const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = { &_htp_burst_power_config.value(), nullptr };
    _htp_perf_infra->setPowerConfig(_htp_power_config_id.value(), cfgs);
}



void QnnBackend::end_burst() {
    if (!burst || !_htp_power_config_id)
        return;

    debug("Switching to normal power mode...");
    const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = { &_htp_normal_power_config.value(), nullptr };
    _htp_perf_infra->setPowerConfig(_htp_power_config_id.value(), cfgs);
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
