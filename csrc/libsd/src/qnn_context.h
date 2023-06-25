#ifndef LIBSD_QNN_CONTEXT_H
#define LIBSD_QNN_CONTEXT_H

#include <list>
#include <string>
#include <memory>

#include <QnnInterface.h>


namespace libsd {

enum class QnnBackend : int {
    CPU,
    GPU,
    DSP,
    HTP,
    HTA
};


class QnnBackendLibrary {
public:
    static std::shared_ptr<QnnBackendLibrary> get(QnnBackend backend);
    virtual ~QnnBackendLibrary();

    void initialize_logs();

    auto get_backend() const { return backend; }

private:
    QnnBackendLibrary(QnnBackend backend);

    QnnBackend backend;
    void* dl = nullptr;
    QNN_INTERFACE_VER_TYPE interface;
    Qnn_LogHandle_t log_hnd;
};


class QnnContext {
public:
    QnnContext(QnnBackend backend, std::list<std::string> const& op_packages = std::list<std::string>(), bool burst = true);

private:
    std::shared_ptr<QnnBackendLibrary> backend_lib;

    Qnn_BackendHandle_t backend_hnd = nullptr;
    QnnBackend_Config_t** backend_cfg = nullptr;
    Qnn_DeviceHandle_t device_hnd = nullptr;
    Qnn_ContextHandle_t context_hnd = nullptr;
    QnnContext_Config_t** context_cfg = nullptr;

    void _init_backend();
    void _init_device();
    void _init_context();
};

}

#endif // LIBSD_QNN_CONTEXT_H
