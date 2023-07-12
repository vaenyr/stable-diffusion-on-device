#include "context.h"
#include "error.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <array>
#include <mutex>
#include <thread>

using namespace libsdod;


Context::Context(std::string const& models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, LogLevel log_level, bool use_htp)
    : models_dir(models_dir), latent_channels(latent_channels), latent_spatial(latent_spatial), upscale_factor(upscale_factor), use_htp(use_htp),
    _random_gen{ std::random_device{}() }, _normal{ 0, 1 } {
    _error_table = allocate_error_table();
    _logger.set_level(log_level);
    if (models_dir.empty())
        this->models_dir = '.';
    else if (models_dir.back() == '/')
        this->models_dir.pop_back();
}


Context::~Context() {
    temb_in.reset();
    temb_out.reset();
    tokens.reset();
    p.reset();
    x.reset();
    t.reset();
    p_cond.reset();
    p_uncond.reset();
    e.reset();
    y.reset();
    img.reset();
    other_tensors.clear();

    _model.reset();
    _tokenizer.reset();
    _solver.reset();
    _qnn_graphs.clear();

    _qnn.reset();
}


void Context::init_mt(unsigned int steps) {
    auto&& tick = std::chrono::high_resolution_clock::now();

    auto&& init_models = std::thread([this, steps]() {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;
        initialize_qnn();
        load_models();
        prepare_buffers();
        prepare_schedule(steps);
    });

    auto&& init_tokenizer = std::thread([this]() {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;
        load_tokenizer();
    });

    auto&& init_solver = std::thread([this]() {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;
        prepare_solver();
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

    _qnn = std::shared_ptr<QnnBackend>(new QnnBackend(use_htp ? QnnBackendType::HTP : QnnBackendType::GPU));
    _qnn_initialized = true;
}


void Context::load_models() {
    if (_failed_and_gave_up)
        return;
    if (!_qnn_initialized)
        return;
    if (_model)
        return;

#if !defined(NOTHREADS) && !defined(LIBSDOD_DEBUG)
    std::map<std::string, QnnGraph*> _graphs;
    std::mutex _graphs_mutex;
    auto&& _graph_names = std::array{ "unet.serialized", "text_encoder.serialized", "vae_decoder.serialized", "temb" };

    auto&& get_model_async = [this, &_graphs, &_graphs_mutex](const char* name) {
        auto&& _log_guard = activate_logger();
        (void)_log_guard;

        std::string filename;
        if (use_htp)
            filename = std::string(name) + ".bin";
        else
            filename = std::string(name) + ".so";

        info("Attempting to load a model: {}", filename);
        auto&& path = models_dir + "/" + filename;
        auto&& graphs = _qnn->load_graphs(path, use_htp);
        if (graphs.empty())
            throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("Deserialized context {} does not contain any graphs!", path), "load_models", __FILE__, STR(__LINE__));
        if (graphs.size() > 1)
            info("Warning: deserialized context {} contains more than 1 graph {}, only the first one will be used", path, graphs.size());

        graphs.front().set_name(name);

        auto&& _guard = std::lock_guard<std::mutex>{ _graphs_mutex };
        (void)_guard;
        info("Model {} loaded", name);
        _graphs[name] = &graphs.front();
        _qnn_graphs.splice(_qnn_graphs.end(), std::move(graphs), graphs.begin());
    };

    std::list<std::thread> _loading_threads;
    for (auto&& gname : _graph_names)
        _loading_threads.emplace_back(get_model_async, gname);

    for (auto&& t : _loading_threads)
        t.join();

    _model.emplace(StableDiffusionModel{
        .unet = *_graphs["unet.serialized"],
        .cond_model = *_graphs["text_encoder.serialized"],
        .decoder = *_graphs["vae_decoder.serialized"],
        .temb = *_graphs["temb"]
    });
#else
    auto&& get_model = [this](const char* name) -> graph_ref {
        std::string filename;
        if (use_htp)
            filename = std::string(name) + ".bin";
        else
            filename = std::string(name) + ".qnn.so";

        info("Attempting to load a model: {}", filename);
        auto&& path = models_dir + "/" + filename;
        auto&& graphs = _qnn->load_graphs(path, use_htp);
        if (graphs.empty())
            throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("Deserialized context {} does not contain any graphs!", path), "load_models", __FILE__, STR(__LINE__));
        if (graphs.size() > 1)
            info("Warning: deserialized context {} contains more than 1 graph {}, only the first one will be used", path, graphs.size());

        graphs.front().set_name(name);
        _qnn_graphs.splice(_qnn_graphs.end(), std::move(graphs), graphs.begin());
        return _qnn_graphs.back();
    };

    _model.emplace(StableDiffusionModel{
        .unet = get_model("unet.serialized"),
        .cond_model = get_model("text_encoder.serialized"),
        .decoder = get_model("vae_decoder.serialized"),
        .temb = get_model("temb")
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
    temb_in.emplace(_model->temb.allocate_input(0));
    temb_out.emplace(_model->temb.allocate_output(0));

    tokens.emplace(_model->cond_model.allocate_input(0));
    p.emplace(_model->cond_model.allocate_output(0));

    x.emplace(_model->unet.allocate_input(0));
    t.emplace(_model->unet.allocate_input(1));
    p_cond.emplace(_model->unet.allocate_input(2));
    p_uncond.emplace(_model->unet.allocate_input(2));
    e.emplace(_model->unet.allocate_output(0));

    y.emplace(_model->decoder.allocate_input(0));
    img.emplace(_model->decoder.allocate_output(0));

    _model->cond_model.verify();
    _model->decoder.verify();
    _model->unet.verify();
    _model->temb.verify();

    p_host.resize(p->get_num_elements(1));
    x_host.resize(latent_channels * latent_spatial * latent_spatial);
    e_host.resize(latent_channels * latent_spatial * latent_spatial);
    img_host.resize(3 * latent_spatial * upscale_factor * latent_spatial * upscale_factor);

    // precompute empty prompt conditioning
    std::vector<Tokenizer::token_type> tokens_host;
    _tokenizer->tokenize(tokens_host, "");
    tokens->set_data(tokens_host);
    _model->cond_model.execute();
    p->get_data(p_host);
    p_uncond->set_data(p_host);

    info("Input/output buffers created and prepared!");
}


void Context::prepare_schedule(unsigned int steps) {
    if (_failed_and_gave_up)
        return;
    if (!_solver)
        return;
    if (steps != 20)
        throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("steps!=20 is currently not implemented, got: {}", steps), __func__, __FILE__, STR(__LINE__));

    std::vector<float> _schedule;
    _solver->prepare(steps, _schedule);

    //compute time embeddings
    constexpr float max_period = 10000.0f;
    constexpr unsigned int mode_dim = 320;
    constexpr unsigned int temb_dim = mode_dim * 4;
    static_assert(mode_dim % 2 == 0, "Odd numbers not handled correctly at the moment, please fix");

    float log_period = -std::log(max_period);
    std::vector<float> mode;
    mode.resize(mode_dim, 0.0f);
    t_embeddings.resize(steps);

    for (auto i : range(steps)) {
        auto half = mode_dim/2;
        t_embeddings[i].resize(temb_dim, 0.0f);
        for (auto j : range(half)) {
            auto arg = _schedule[i] * std::exp(log_period * j / half);
            mode[j] = std::cos(arg);
            mode[half+j] = std::sin(arg);
        }

        temb_in->set_data(mode);
        _model->temb.execute();
        temb_out->get_data(t_embeddings[i]);
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

    // auto&& data_preview = [](std::vector<float> const& data, std::string const& msg) {
    //     std::vector<float> copy(data.data(), data.data() + std::min<size_t>(data.size(), 15));
    //     error("{}: (size: {}) {}", msg, data.size(), copy);
    // };

    auto&& burst_scope_guard = scope_guard([this](){ _qnn->start_burst(); }, [this]() { _qnn->end_burst(); });
    (void)burst_scope_guard;

    auto&& tick = std::chrono::high_resolution_clock::now();
    auto&& tokens_host = _tokenizer->tokenize(prompt);
    tokens->set_data(tokens_host);
    _model->cond_model.execute();
    p->get_data(p_host);
    p_cond->set_data(p_host);
    auto&& tock = std::chrono::high_resolution_clock::now();
    _report_time("Conditioning", tick, tock);

    for (auto& f : x_host)
        f = _normal(_random_gen);

    // //debug
    // tmp.resize(p_cond->get_num_elements(1));
    // p->get_data(tmp);
    // data_preview(tmp, "Prompt embedding");

    unsigned int step = 0;
    for (auto&& t_host : t_embeddings) {
        tick = std::chrono::high_resolution_clock::now();

        // data_preview(x_host, format("Unet input, step: {}", step));
        // data_preview(t_host, format("Unet t, step: {}", step));

        t->set_data(t_host);
        x->set_data(x_host);
        p_cond->activate();

        _model->unet.execute();

        // //debug
        // tmp.resize(e->get_num_elements(1));
        // e->get_data(tmp);
        // data_preview(tmp, "    Cond output");

        if (guidance == 1.0f)
            e->get_data(e_host);
        else {
            e->get_data(e_host, guidance);

            p_uncond->activate();

            _model->unet.execute();

            // //debug
            // tmp.resize(e->get_num_elements(1));
            // e->get_data(tmp);
            // data_preview(tmp, "    Uncond output");

            e->get_data(e_host, 1-guidance, true);

            // data_preview(e_host, format("Unet output, step: {}", step));
        }

        _solver->update(step++, x_host, e_host);

        tock = std::chrono::high_resolution_clock::now();
        _report_time("Single iteration", tick, tock);
    }

    tick = std::chrono::high_resolution_clock::now();

    y->set_data(x_host);
    _model->decoder.execute();
    img->get_data(img_host); //, 1 / 0.18215, false);
    debug("Output image has {} elements", img_host.size());
    auto* output_ptr = output.data_ptr();
    // decode img to uint8 pixels
    for (auto i : range(img_host.size())) {
        auto f = img_host[i];
        output_ptr[i] = static_cast<uint8_t>(std::clamp(255 * f, 0.0f, 255.0f));
    }

    tock = std::chrono::high_resolution_clock::now();
    _report_time("Decoding", tick, tock);

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
