#include "context.h"
#include "error.h"
#include "utils.h"

using namespace libsd;


Context::Context(std::string const& models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, LogLevel log_level)
    : models_dir(models_dir), latent_channels(latent_channels), latent_spatial(latent_spatial), upscale_factor(upscale_factor) {
    _error_table = allocate_error_table();
    _logger.set_level(log_level);
}


Context::~Context() {
}


void Context::initialize_qnn() {
    if (_failed_and_gave_up)
        return;
    if (_qnn_initialized)
        return;

    _qnn = std::shared_ptr<QnnBackend>(new QnnBackend(QnnBackendType::HTP));
    _qnn_initialized = true;
}


void Context::load_models() {
    if (_failed_and_gave_up)
        return;
    if (!_qnn_initialized)
        return;
    if (_model)
        return;

    auto&& get_model = [this](const char* name) -> graph_ref {
        auto&& path = models_dir + "/" + name;
        auto&& graphs = _qnn->load_context(path);
        if (graphs.empty())
            throw libsd_exception(ErrorCode::INVALID_ARGUMENT, format("Deserialized context {} does not contain any graphs!", path), "load_models", __FILE__, STR(__LINE__));
        if (graphs.size() > 1)
            info("Warning: deserialized context {} contains more than 1 graph {}, only the first one will be used", path, graphs.size());
        return graphs.front();
    };

    _model.emplace(StableDiffusionModel{
        .cond_model = get_model("cond_model.bin"),
        .decoder = get_model("decoder.bin"),
        .unet_inputs = get_model("sd_unet_inputs.bin"),
        .unet_middle = get_model("sd_unet_middle.bin"),
        .unet_outputs = get_model("sd_unet_outputs.bin"),
        .unet_head = get_model("sd_unet_head.bin")
    });

    info("All models loaded!");
}


void Context::prepare_solver() {
    info("ODE solver prepared!");
}


void Context::prepare_buffers() {
    info("Input/output buffers created and prepared!");
}


void Context::prepare_schedule(unsigned int steps) {
    info("Time schedule prepared for {} steps!", steps);
}


void Context::generate(std::string const& prompt, float guidance, Buffer<unsigned char>& output) {
    if (_failed_and_gave_up)
        return;
    if (!_qnn_initialized)
        return;
    if (!_model)
        return;

    info("Starting image generation for prompt: \"{}\" and guidance {}", prompt, guidance);
    debug("Current steps: {}", t_schedule.size());

    // _tokenize(prompt);

    // _set_inputs<0>(_model.cond_model, prompt_tokens);
    // _run_model(_model.cond_model);


    x_curr.resize(latent_channels * latent_spatial * latent_spatial);
    y_cond.resize(latent_channels * latent_spatial * latent_spatial);
    if (guidance != 1.0f)
        y_cond.resize(latent_channels * latent_spatial * latent_spatial);

    // _draw_random_sample(x_curr);

    // for (auto&& t : t_schedule) {
    //     _set_inputs<0, 1>(_model.unet_inputs, x_curr, t);
    //     _set_inputs<1>(_model.unet_middle, t);
    //     _set_inputs<1>(_model.unet_middle, t);
    //     _set_inputs<16>(_model.unet_middle, t);
    //     _set_inputs<1>(_model.unet_middle, t);

    //     _run_model(_model.unet_inputs);
    //     _run_model(_model.unet_middle);
    //     _run_model(_model.unet_outputs);
    //     _run_model(_model.unet_head);

    //     if (guidance != 1.0f) {
    //         _set_inputs<
    //     }
    // }

    // _run_model(_model.decoder);

    info("Image successfully generated!");
}


Buffer<unsigned char> Context::allocate_output() const {
    std::size_t required_len = 3 * latent_spatial * latent_spatial * upscale_factor * upscale_factor;
    return Buffer<unsigned char>(required_len);
}


Buffer<unsigned char> Context::reuse_buffer(unsigned char* buffer, unsigned int buffer_len) const {
    if (!buffer)
        throw libsd_exception(ErrorCode::INVALID_ARGUMENT, "Asked to reuse a nullptr buffer", __func__, __FILE__, STR(__LINE__));

    std::size_t required_len = 3 * latent_spatial * latent_spatial * upscale_factor * upscale_factor;
    if (buffer_len < required_len)
        throw libsd_exception(ErrorCode::INVALID_ARGUMENT, "Provided buffer is too small, missing " + std::to_string(required_len - buffer_len) + " bytes", __func__, __FILE__, STR(__LINE__));

    return Buffer<unsigned char>(buffer, buffer_len);
}
