#ifndef LIBSD_QNN_CONTEXT_H
#define LIBSD_QNN_CONTEXT_H

#include <list>
#include <span>
#include <vector>
#include <string>
#include <memory>
#include <optional>

#include <QnnInterface.h>
#include <System/QnnSystemInterface.h>


namespace libsd {

class QnnApi;
class QnnTensor;
class QnnGraph;
class QnnContext;
class QnnBackend;


template <class T>
using qnn_hnd = std::shared_ptr<std::remove_pointer_t<T>>;
using graph_ref = std::add_lvalue_reference_t<QnnGraph>;
using graph_refs = std::list<std::reference_wrapper<QnnGraph>>;


#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

typedef void* (*RpcMemAllocFn_t)(int, uint32_t, int);
typedef void (*RpcMemFreeFn_t)(void*);
typedef int (*RpcMemToFdFn_t)(void*);


enum class QnnBackendType : int {
    CPU,
    GPU,
    DSP,
    HTP,
    HTA
};


class QnnApi {
public:
    static std::shared_ptr<QnnApi> get(QnnBackendType backend);
    virtual ~QnnApi();

    auto get_backend_type() const { return backend; }

    qnn_hnd<Qnn_BackendHandle_t> create_backend(const QnnBackend_Config_t** cfg) const;
    qnn_hnd<Qnn_DeviceHandle_t> create_device(const QnnDevice_Config_t** cfg) const;
    qnn_hnd<Qnn_ContextHandle_t> create_context(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** cfg) const;
    qnn_hnd<Qnn_ContextHandle_t> create_context(std::vector<unsigned char> const& buffer, Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** cfg) const;

    void register_op_package(std::string const& package_path, std::string const& package_interface_provider) const;

    QnnSystemContext_BinaryInfo_t const& get_binary_info(std::vector<unsigned char>& buffer) const;
    Qnn_GraphHandle_t retrieve_graph(Qnn_ContextHandle_t context, const char* graph_name) const;

    std::pair<std::shared_ptr<void>,int> allocate_ion(uint32_t size);
    qnn_hnd<Qnn_MemHandle_t> mem_register(Qnn_ContextHandle_t ctx, Qnn_MemDescriptor_t desc);

    bool has_ion() const { return bool(cdsp_dl); }

private:
    QnnApi(QnnBackendType backend);

    QnnBackendType backend;
    std::shared_ptr<void> dl;
    std::shared_ptr<void> system_dl;
    std::shared_ptr<void> cdsp_dl ;
    qnn_hnd<Qnn_LogHandle_t> log_hnd;
    qnn_hnd<QnnSystemContext_Handle_t> system_hnd;

    QNN_INTERFACE_VER_TYPE interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE system_interface;
    RpcMemAllocFn_t rpcmem_alloc = nullptr;
    RpcMemFreeFn_t rpcmem_free = nullptr;
    RpcMemToFdFn_t rpcmem_to_fd = nullptr;
};


class QnnTensor {
    friend class QnnGraph;
public:
    QnnTensor(QnnTensor&& other) = default;
    QnnTensor(QnnTensor const& ohter) = delete;

    void activate() const; // make sure the data associated with this object is attached to the target slot

    uint32_t get_num_elements(unsigned int batch_size) const { return get_num_elements(target, batch_size); }
    uint8_t get_element_size() const { return get_element_size(target); }

    static uint32_t get_num_elements(Qnn_Tensor_t const& t, unsigned int batch_size=1);
    static uint8_t get_element_size(Qnn_Tensor_t const& t);

private:
    QnnTensor(QnnApi& api, Qnn_ContextHandle_t ctx, Qnn_Tensor_t& target, unsigned int batch_size=1); //allocate new
    QnnTensor(QnnTensor const& other, Qnn_Tensor_t& target); //reuse the same allocation for different input/output slot

    bool is_ion = false;
    unsigned int batch_size;

    std::shared_ptr<void> data;
    uint32_t data_size = 0;
    int data_fd = -1;
    qnn_hnd<Qnn_MemHandle_t> data_hnd;

    Qnn_Tensor_t& target;
};


class QnnGraph {
    friend class QnnBackend;
public:
    QnnTensor allocate_input(unsigned int idx, unsigned batch=1);
    QnnTensor attach_input(unsigned int idx, QnnTensor const& t);

    QnnTensor allocate_output(unsigned int idx, unsigned batch=1);
    QnnTensor attach_output(unsigned int idx, QnnTensor const& t);

private:
    QnnGraph(qnn_hnd<Qnn_ContextHandle_t> ctx, std::shared_ptr<QnnApi> api, const char* name, Qnn_Tensor_t* inputs, unsigned int num_inputs,
        Qnn_Tensor_t* outputs, unsigned int num_outputs, Qnn_GraphHandle_t graph);

    const char* name;
    std::span<Qnn_Tensor_t> inputs;
    std::span<Qnn_Tensor_t> outputs;

    std::vector<std::optional<QnnTensor>> inputs_mem;
    std::vector<std::optional<QnnTensor>> outputs_mem;

    Qnn_GraphHandle_t graph;
    qnn_hnd<Qnn_ContextHandle_t>::weak_type ctx;
    std::shared_ptr<QnnApi> api;
};


class QnnContext {
    friend class QnnBackend;
public:
    QnnContext(QnnContext const& other) = delete;
    QnnContext(QnnContext&& other) = default;

private:
    QnnContext(qnn_hnd<Qnn_ContextHandle_t> ctx, std::vector<QnnGraph>&& graphs);

    qnn_hnd<Qnn_ContextHandle_t> ctx;
    std::vector<QnnGraph> graphs;
};



class QnnBackend {
public:
    QnnBackend(QnnBackendType backend, std::list<std::string> const& op_packages = std::list<std::string>(), bool burst = true);

    graph_refs load_context(std::string const& context_blob);
    graph_refs load_model(std::string const& model_so);

    // tensor_refs run(graph_ref graph, tensor_refs& inputs);

private:
    std::shared_ptr<QnnApi> api;

    qnn_hnd<Qnn_BackendHandle_t> backend_hnd;
    qnn_hnd<Qnn_DeviceHandle_t> device_hnd;

    std::list<QnnContext> ctx;

    void _init_backend();
    void _init_device();
    void _init_context();
};

}

#endif // LIBSD_QNN_CONTEXT_H
