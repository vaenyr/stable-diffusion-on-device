#ifndef LIBSD_QNN_CONTEXT_H
#define LIBSD_QNN_CONTEXT_H

#include <list>
#include <span>
#include <string>
#include <memory>

#include <QnnInterface.h>


namespace libsd {

template <class T>
using qnn_hnd = std::shared_ptr<std::remove_pointer_t<T>>;

using graph_ref = std::add_lvalue_reference_t<_QnnGraph>;
using graph_refs = std::list<graph_ref>;

#ifdef __ANDROID__
using tensor_ref = ion_hnd;
#else
using tensor_ref = void*;
#endif

using tensor_refs = std::list<tensor_ref>;


enum class QnnBackend : int {
    CPU,
    GPU,
    DSP,
    HTP,
    HTA
};


class _QnnGraph {
    friend class QnnApi;
    friend class QnnContext;
private:
    const char* name;
    std::span<Qnn_Tensor_t> inputs;
    std::span<Qnn_Tensor_t> outputs;

    qnn_hnd<Qnn_GraphHandle_t> graph;
};


class _QnnContext {
    friend class QnnApi;
    friend class QnnContext;
public:
    _QnnContext(_QnnContext const& other) = delete;
    _QnnContext(_QnnContext&& other);

    auto&& begin() const { return graphs.begin(); }
    auto&& end() const { return graphs.end(); }
    auto&& get_graph_info(unsigned int idx) const { return graphs[idx]; }
    auto&& get_num_graphs() const { return graphs.size(); }

    void add_graph(_QnnGraph&& g) { graphs.emplace_back(std::move(g)); }

private:
    _QnnContext(qnn_hnd<Qnn_ContextHandle_t> ctx, std::vector<_QnnGraph>&& graphs);

    qnn_hnd<Qnn_ContextHandle_t> ctx;
    std::vector<_QnnGraph> graphs;
};


class QnnApi {
public:
    static std::shared_ptr<QnnApi> get(QnnBackend backend);
    virtual ~QnnApi();

    auto get_backend_type() const { return backend; }

    qnn_hnd<Qnn_BackendHandle_t> create_backend(QnnBackend_Config_t** cfg) const;
    qnn_hnd<Qnn_DeviceHandle_t> create_device(QnnDevice_Config_t** cfg) const;
    qnn_hnd<Qnn_ContextHandle_t> create_context(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, QnnContext_Config_t** cfg) const;
    qnn_hnd<Qnn_ContextHandle_t> create_context(std::vector<unsigned char> const& buffer, Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, QnnContext_Config_t** cfg) const;

    void register_op_package(std::string const& package_path, std::string const& package_interface_provider) const;

    std::span<QnnSystemContext_BinaryInfo_t*> get_binary_info() const;

private:
    QnnApi(QnnBackend backend);

    QnnBackend backend;
    std::unique_ptr<void*> dl = nullptr;
    std::unique_ptr<void*> system_dl = nullptr;
    qnn_hnd<Qnn_LogHandle_t> log_hnd = nullptr;
    qnn_hnd<QnnSystemContext_Handle_t> system_hnd = nullptr;

    QNN_INTERFACE_VER_TYPE interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE system_interface;
};


class QnnContext {
public:
    QnnContext(QnnBackend backend, std::list<std::string> const& op_packages = std::list<std::string>(), bool burst = true);

    graph_refs load_context(std::string const& context_blob);
    graph_refs load_model(std::string const& model_so);

    tensor_refs run(graph_ref graph, tensor_refs& inputs);

private:
    std::shared_ptr<QnnApi> api;

    qnn_hnd<Qnn_BackendHandle_t> backend_hnd;
    qnn_hnd<Qnn_DeviceHandle_t> device_hnd;

    std::list<_QnnContext> ctx;

    void _init_backend();
    void _init_device();
    void _init_context();
};

}

#endif // LIBSD_QNN_CONTEXT_H
