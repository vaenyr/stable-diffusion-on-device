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
    _qnn = std::shared_ptr<QnnContext>(new QnnContext(QnnBackend::HTP));
    _qnn_initialized = true;
}

void Context::load_models() {
    if (_failed_and_gave_up)
        return;
    if (!_qnn_initialized)
        return;
    if (_models_loaded)
        return;

    _models_loaded = true;
}

void Context::prepare_sampler() {
}

void Context::generate(std::string const& prompt, float guidance, Buffer<unsigned char>& output) {
    if (_failed_and_gave_up)
        return;
    if (!_qnn_initialized)
        return;
    if (!_models_loaded)
        return;

    float* x_curr = new float[latent_channels * latent_spatial * latent_spatial];
    float* x_next = new float[latent_channels * latent_spatial * latent_spatial];
    float* uncond = nullptr;
    if (guidance != 1.0f)
        uncond = new float[latent_channels * latent_spatial * latent_spatial];

    if (uncond)
        delete[] uncond;
    delete[] x_curr;
    delete[] x_next;
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
