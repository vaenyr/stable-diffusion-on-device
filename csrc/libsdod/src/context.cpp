#include "context.h"
#include "error.h"
#include "utils.h"

#include <chrono>
#include <cmath>

using namespace libsdod;


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

    auto&& get_model = [this](const char* name, const char* gname) -> graph_ref {
        info("Attempting to load a model: {}", name);
        auto&& path = models_dir + "/" + name;
        auto&& graphs = _qnn->load_context(path);
        if (graphs.empty())
            throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("Deserialized context {} does not contain any graphs!", path), "load_models", __FILE__, STR(__LINE__));
        if (graphs.size() > 1)
            info("Warning: deserialized context {} contains more than 1 graph {}, only the first one will be used", path, graphs.size());
        auto&& ret = graphs.front();
        ret.get().set_name(gname);
        return ret;
    };

    _model.emplace(StableDiffusionModel{
        .unet_outputs = get_model("sd_unet_outputs.bin", "unet_outputs"),
        .unet_inputs = get_model("sd_unet_inputs.bin", "unet_inputs"),
        .cond_model = get_model("cond_model.bin", "cond"),
        .unet_middle = get_model("sd_unet_middle.bin", "unet_middle"),
        .decoder = get_model("decoder.bin", "decoder"),
        .unet_head = get_model("sd_unet_head.bin", "unet_middle")
    });

    info("All models loaded!");
}


void Context::prepare_solver() {
    info("ODE solver prepared!");
}


void Context::prepare_buffers() {
    if (!_model)
        return;

    // allocate important tensors
    tokens.emplace(_model->cond_model.allocate_input(0));
    auto& p = other_tensors.emplace_back(_model->cond_model.allocate_output(0));
    x.emplace(_model->unet_inputs.allocate_input(0));
    t.emplace(_model->unet_inputs.allocate_input(1));
    y.emplace(_model->unet_head.allocate_output(0));
    img.emplace(_model->decoder.allocate_output(0));

    // attach time embeddings
    other_tensors.emplace_back(_model->unet_middle.attach_input(1, t.value()));
    other_tensors.emplace_back(_model->unet_outputs.attach_input(1, t.value()));

    // attach prompt
    other_tensors.emplace_back(_model->unet_inputs.attach_input(2, p, true, false)); // TODO: inputs seem to be transposed... for now ignore
    other_tensors.emplace_back(_model->unet_middle.attach_input(2, p, true, false));
    other_tensors.emplace_back(_model->unet_outputs.attach_input(2, p, true, false));

    // cond model, other output (TODO: get rid of? seems unused)
    other_tensors.emplace_back(_model->cond_model.allocate_output(1));

    // inputs <-> { middle, outputs }
    for (auto i : range(_model->unet_inputs.get_num_outputs())) {
        auto& o = other_tensors.emplace_back(_model->unet_inputs.allocate_output(i));
        //skip connection to outputs
        other_tensors.emplace_back(_model->unet_outputs.attach_input(3+i, o));
        //if last output, also connect to middle
        if (i+1 == _model->unet_inputs.get_num_outputs())
            other_tensors.emplace_back(_model->unet_middle.attach_input(0, o));
    }

    // middle <-> outputs
    other_tensors.emplace_back(_model->unet_middle.allocate_output(0));
    other_tensors.emplace_back(_model->unet_outputs.attach_input(0, other_tensors.back()));

    // outputs <-> head
    other_tensors.emplace_back(_model->unet_outputs.allocate_output(0));
    other_tensors.emplace_back(_model->unet_head.attach_input(0, other_tensors.back()));

    // head <-> decoder
    other_tensors.emplace_back(_model->decoder.attach_input(0, y.value()));

    _model->cond_model.verify();
    _model->decoder.verify();
    _model->unet_inputs.verify();
    _model->unet_middle.verify();
    _model->unet_outputs.verify();
    _model->unet_head.verify();

    tokens_host.resize(77);
    x_host.resize(latent_channels * latent_spatial * latent_spatial);
    y_host.resize(latent_channels * latent_spatial * latent_spatial);
    img_host.resize(3 * latent_spatial * upscale_factor * latent_spatial * upscale_factor);

    info("Input/output buffers created and prepared!");
}


void Context::prepare_schedule(unsigned int steps) {
    if (steps != 20)
        throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("steps!=20 is currently not implemented, got: {}", steps), __func__, __FILE__, STR(__LINE__));

    //TODO: compute schedule properly, this is taken from plms.py with 20 ddim steps
    std::vector<uint32_t> _schedule = { 951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101, 51, 1 };

    float log_period = -std::log(10000.0f);

    //compute time embeddings
    t_embeddings.resize(steps);
    for (auto i : range(steps)) {
        auto half = unet_dim/2;
        t_embeddings[i].resize(unet_dim, 0.0f);
        for (auto j : range(half)) {
            auto arg = _schedule[i] * (std::exp(log_period * j) / half);
            t_embeddings[i][j] = std::cos(arg);
            t_embeddings[i][half+j] = std::sin(arg);
        }
    }

    info("Time schedule prepared for {} steps!", steps);
}


void Context::generate(std::string const& prompt, float guidance, Buffer<unsigned char>& output) {
    if (_failed_and_gave_up)
        return;
    if (!_qnn_initialized)
        return;
    if (!_model)
        return;

    auto&& start = std::chrono::high_resolution_clock::now();

    info("Starting image generation for prompt: \"{}\" and guidance {}", prompt, guidance);
    debug("Current steps: {}", t_embeddings.size());

    auto&& _report_time = [](const char* name,
        std::chrono::high_resolution_clock::time_point const& t1,
        std::chrono::high_resolution_clock::time_point const& t2) {
        auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        info("{} took {}ms", name, diff.count());
    };

    auto&& burst_scope_guard = scope_guard([this](){ _qnn->start_burst(); }, [this]() { _qnn->end_burst(); });
    (void)burst_scope_guard;

    tokens->set_data(tokens_host);

    auto&& tick = std::chrono::high_resolution_clock::now();
    _model->cond_model.execute();
    auto&& tock = std::chrono::high_resolution_clock::now();
    _report_time("Conditioning", tick, tock);

    for (auto&& t_host : t_embeddings) {
        tick = std::chrono::high_resolution_clock::now();

        t->set_data(t_host);
        x->set_data(x_host);

        _model->unet_inputs.execute();
        _model->unet_middle.execute();
        _model->unet_outputs.execute();
        _model->unet_head.execute();

        if (guidance == 1.0f)
            y->get_data(y_host);
        else {
            y->get_data(y_host, guidance);

            _model->unet_inputs.execute();
            _model->unet_middle.execute();
            _model->unet_outputs.execute();
            _model->unet_head.execute();

            y->get_data(y_host, 1-guidance);
        }

        // TODO: ODE solver step, should write to x_host

        tock = std::chrono::high_resolution_clock::now();
        _report_time("Single iteration", tick, tock);
    }

    y->set_data(x_host);

    tick = std::chrono::high_resolution_clock::now();
    _model->decoder.execute();
    tock = std::chrono::high_resolution_clock::now();
    _report_time("Decoding", tick, tock);

    img->get_data(img_host);

    auto* output_ptr = output.data_ptr();
    // decode img to uint8 pixels
    for (auto i : range(img_host.size())) {
        auto f = img_host[i];
        output_ptr[i] = static_cast<uint8_t>(255 * std::clamp(((f + 1) / 2), 0.0f, 1.0f));
    }


    info("Image successfully generated!");
    auto&& end = std::chrono::high_resolution_clock::now();
    _report_time("Image generation", start, end);
}


Buffer<unsigned char> Context::allocate_output() const {
    std::size_t required_len = 3 * latent_spatial * latent_spatial * upscale_factor * upscale_factor;
    return Buffer<unsigned char>(required_len);
}


Buffer<unsigned char> Context::reuse_buffer(unsigned char* buffer, unsigned int buffer_len) const {
    if (!buffer)
        throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, "Asked to reuse a nullptr buffer", __func__, __FILE__, STR(__LINE__));

    std::size_t required_len = 3 * latent_spatial * latent_spatial * upscale_factor * upscale_factor;
    if (buffer_len < required_len)
        throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, "Provided buffer is too small, missing " + std::to_string(required_len - buffer_len) + " bytes", __func__, __FILE__, STR(__LINE__));

    return Buffer<unsigned char>(buffer, buffer_len);
}
