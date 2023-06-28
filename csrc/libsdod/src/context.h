#ifndef LIBSDOD_CONTEXT_H
#define LIBSDOD_CONTEXT_H

#include <string>
#include <optional>

#include "errors.h"
#include "buffer.h"
#include "qnn_context.h"
#include "logging.h"


namespace libsdod {


struct StableDiffusionModel {
    graph_ref unet_outputs;
    graph_ref unet_inputs;
    graph_ref cond_model;
    graph_ref unet_middle;
    graph_ref decoder;
    graph_ref unet_head;
};


class Context {
public:
    Context(std::string const& models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, LogLevel log_level);
    virtual ~Context();

    void initialize_qnn();
    void load_models();
    void prepare_solver();
    void prepare_buffers();
    void prepare_schedule(unsigned int steps);

    void generate(std::string const& prompt, float guidance, Buffer<unsigned char>& output);

    ErrorTable get_error_table() const { return _error_table; }

    Buffer<unsigned char> allocate_output() const;
    Buffer<unsigned char> reuse_buffer(unsigned char* buffer, unsigned int buffer_len) const;

    Logger& get_logger() { return _logger; }
    Logger const& get_logger() const { return _logger; }
    ActiveLoggerScopeGuard activate_logger() { return ActiveLoggerScopeGuard(_logger); }

private:
    std::string models_dir;
    unsigned int latent_channels;
    unsigned int latent_spatial;
    unsigned int upscale_factor;

    bool _failed_and_gave_up = false;
    bool _qnn_initialized = false;

    ErrorTable _error_table;
    Logger _logger;

    std::shared_ptr<QnnBackend> _qnn;
    std::optional<StableDiffusionModel> _model;

    std::vector<unsigned int> tokens_host;
    std::vector<float> x_host;
    std::vector<float> y_host;
    std::vector<float> img_host;

    std::vector<std::vector<float>> t_embeddings; // sequence of encoded timesteps

    unsigned int unet_dim = 1280; // TODO: expose as arg?

    std::optional<QnnTensor> tokens;
    std::optional<QnnTensor> x;
    std::optional<QnnTensor> t;
    std::optional<QnnTensor> y;
    std::optional<QnnTensor> img;

    tensor_list other_tensors;
};

}

#endif // LIBSDOD_CONTEXT_H
