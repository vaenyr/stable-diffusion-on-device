#include "context.h"
#include "error.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <array>
#include <mutex>
#include <thread>

using namespace libsdod;


Context::Context(std::string const& models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, LogLevel log_level)
    : models_dir(models_dir), latent_channels(latent_channels), latent_spatial(latent_spatial), upscale_factor(upscale_factor),
    _random_gen{ std::random_device{}() }, _normal{ 0, 1 } {
    _error_table = allocate_error_table();
    _logger.set_level(log_level);
    if (models_dir.empty())
        this->models_dir = '.';
    else if (models_dir.back() == '/')
        this->models_dir.pop_back();
}


Context::~Context() {
}


void Context::init_mt(unsigned int steps) {
    auto&& tick = std::chrono::high_resolution_clock::now();

    auto&& init_models = std::thread([this]() {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;
        initialize_qnn();
        load_models();
        prepare_buffers();
    });

    auto&& init_tokenizer = std::thread([this]() {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;
        load_tokenizer();
    });

    auto&& init_solver = std::thread([this, steps]() {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;
        prepare_solver();
        prepare_schedule(steps);
    });

    init_models.join();
    init_tokenizer.join();
    init_solver.join();

    auto&& tock = std::chrono::high_resolution_clock::now();
    auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    info("Initialization took {}ms", diff.count());
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

#ifndef NOTHREADS
    std::map<std::string, QnnGraph*> _graphs;
    std::mutex _graphs_mutex;
    auto&& _graph_names = std::array{ "sd_unet_outputs", "sd_unet_inputs", "cond_model", "sd_unet_middle", "decoder", "sd_unet_head" };

    auto&& get_model_async = [this, &_graphs, &_graphs_mutex](const char* name) {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;
        std::string filename = std::string(name) + ".bin";
        info("Attempting to load a model: {}", filename);
        auto&& path = models_dir + "/" + filename;
        auto&& graphs = _qnn->load_context(path);
        if (graphs.empty())
            throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("Deserialized context {} does not contain any graphs!", path), "load_models", __FILE__, STR(__LINE__));
        if (graphs.size() > 1)
            info("Warning: deserialized context {} contains more than 1 graph {}, only the first one will be used", path, graphs.size());
        auto&& ret = graphs.front();
        ret.get().set_name(name);
        auto&& _guard = std::lock_guard<std::mutex>{ _graphs_mutex };
        (void)_guard;
        info("Model {} loaded", name);
        _graphs[name] = &ret.get();
    };

    std::list<std::thread> _loading_threads;
    for (auto&& gname : _graph_names)
        _loading_threads.emplace_back(get_model_async, gname);

    for (auto&& t : _loading_threads)
        t.join();

    _model.emplace(StableDiffusionModel{
        .unet_outputs = *_graphs["sd_unet_outputs"],
        .unet_inputs = *_graphs["sd_unet_inputs"],
        .cond_model = *_graphs["cond_model"],
        .unet_middle = *_graphs["sd_unet_middle"],
        .decoder = *_graphs["decoder"],
        .unet_head = *_graphs["sd_unet_head"]
    });
#else
    auto&& get_model = [this](const char* name) -> graph_ref {
        std::string filename = std::string(name) + ".bin";
        info("Attempting to load a model: {}", filename);
        auto&& path = models_dir + "/" + filename;
        auto&& graphs = _qnn->load_context(path);
        if (graphs.empty())
            throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("Deserialized context {} does not contain any graphs!", path), "load_models", __FILE__, STR(__LINE__));
        if (graphs.size() > 1)
            info("Warning: deserialized context {} contains more than 1 graph {}, only the first one will be used", path, graphs.size());
        auto&& ret = graphs.front();
        ret.get().set_name(name);
        return ret;
    };

    _model.emplace(StableDiffusionModel{
        .unet_outputs = get_model("sd_unet_outputs"),
        .unet_inputs = get_model("sd_unet_inputs"),
        .cond_model = get_model("cond_model"),
        .unet_middle = get_model("sd_unet_middle"),
        .decoder = get_model("decoder"),
        .unet_head = get_model("sd_unet_head")
    });
#endif

    info("All models loaded!");
}


void Context::load_tokenizer() {
    if (_failed_and_gave_up)
        return;
    if (_tokenizer)
        return;

    _tokenizer.emplace(models_dir + "/ctokenizer.txt");
    info("Tokenizer created!");
}


void Context::prepare_solver() {
    if (_failed_and_gave_up)
        return;
    if (_solver)
        return;
    _solver.emplace(1000, 0.00085, 0.0120);
    info("ODE solver prepared!");
}


void Context::prepare_buffers() {
    if (_failed_and_gave_up)
        return;
    if (!_model)
        return;

    // allocate important tensors
    tokens.emplace(_model->cond_model.allocate_input(0));
    p_cond.emplace(_model->cond_model.allocate_output(0));
    p_uncond.emplace(_model->cond_model.allocate_output(0));
    x.emplace(_model->unet_inputs.allocate_input(0));
    t.emplace(_model->unet_inputs.allocate_input(1));
    y.emplace(_model->unet_head.allocate_output(0));
    img.emplace(_model->decoder.allocate_output(0));

    // attach time embeddings
    other_tensors.emplace_back(_model->unet_middle.attach_input(1, t.value()));
    other_tensors.emplace_back(_model->unet_outputs.attach_input(1, t.value()));

    // attach prompt
    p_cond_inputs.emplace_back(_model->unet_inputs.attach_input(2, p_cond.value(), true, false)); // TODO: inputs seem to be transposed... for now ignore
    p_cond_inputs.emplace_back(_model->unet_middle.attach_input(2, p_cond.value(), true, false));
    p_cond_inputs.emplace_back(_model->unet_outputs.attach_input(2, p_cond.value(), true, false));

    p_uncond_inputs.emplace_back(_model->unet_inputs.attach_input(2, p_uncond.value(), true, false)); // TODO: inputs seem to be transposed... for now ignore
    p_uncond_inputs.emplace_back(_model->unet_middle.attach_input(2, p_uncond.value(), true, false));
    p_uncond_inputs.emplace_back(_model->unet_outputs.attach_input(2, p_uncond.value(), true, false));

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
    empty_prompt_host.resize(77);
    x_host.resize(latent_channels * latent_spatial * latent_spatial);
    y_host.resize(latent_channels * latent_spatial * latent_spatial);
    img_host.resize(3 * latent_spatial * upscale_factor * latent_spatial * upscale_factor);

    // precompute empty prompt conditioning
    _tokenizer->tokenize(empty_prompt_host, "");
    tokens->set_data(empty_prompt_host);
    p_uncond->activate();
    _model->cond_model.execute();

    p_cond->activate();
    for (auto&& t : p_cond_inputs)
        t.activate();

    info("Input/output buffers created and prepared!");
}


void Context::prepare_schedule(unsigned int steps) {
    if (_failed_and_gave_up)
        return;
    if (!_solver)
        return;
    if (steps != 20)
        throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("steps!=20 is currently not implemented, got: {}", steps), __func__, __FILE__, STR(__LINE__));

    //TODO: compute schedule properly, this is taken from plms.py with 20 ddim steps
    std::vector<uint32_t> _schedule;
    _solver->prepare(steps, _schedule);

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


void Context::set_seed(unsigned int seed) {
    info("Using seed: {}", seed);
    _normal.reset();
    _random_gen.seed(seed);
}


void Context::generate(std::string const& prompt, float guidance, Buffer<unsigned char>& output) {
    if (_failed_and_gave_up)
        return;
    if (!_qnn_initialized)
        return;
    if (!_model)
        return;
    if (!_solver)
        return;
    if (!_tokenizer)
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

    tokens_host = _tokenizer->tokenize(prompt);

    auto&& burst_scope_guard = scope_guard([this](){ _qnn->start_burst(); }, [this]() { _qnn->end_burst(); });
    (void)burst_scope_guard;

    tokens->set_data(tokens_host);

    auto&& tick = std::chrono::high_resolution_clock::now();
    _model->cond_model.execute();
    auto&& tock = std::chrono::high_resolution_clock::now();
    _report_time("Conditioning", tick, tock);

    for (auto& f : x_host)
        f = _normal(_random_gen);

    unsigned int step = 0;
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

            for (auto&& t : p_uncond_inputs)
                t.activate();

            _model->unet_inputs.execute();
            _model->unet_middle.execute();
            _model->unet_outputs.execute();
            _model->unet_head.execute();

            y->get_data(y_host, 1-guidance);

            for (auto&& t : p_cond_inputs)
                t.activate();
        }

        _solver->update(step++, x_host, y_host);

        tock = std::chrono::high_resolution_clock::now();
        _report_time("Single iteration", tick, tock);
    }

    y->set_data(x_host);

    tick = std::chrono::high_resolution_clock::now();
    _model->decoder.execute();
    tock = std::chrono::high_resolution_clock::now();
    _report_time("Decoding", tick, tock);

    img->get_data(img_host, 1 / 0.18215);

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
