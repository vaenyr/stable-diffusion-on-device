#ifndef LIBSDOD_CONTEXT_H
#define LIBSDOD_CONTEXT_H

#include <string>
#include <optional>
#include <random>

#include "errors.h"
#include "buffer.h"
#include "qnn_context.h"
#include "logging.h"
#include "dpm_solver.h"
#include "tokenizer.h"


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

    void init_mt(unsigned int steps);

    void initialize_qnn();
    void load_models();
    void load_tokenizer();
    void prepare_solver();
    void prepare_buffers();
    void prepare_schedule(unsigned int steps);

    void set_seed(unsigned int seed);

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

    std::mt19937 _random_gen;
    std::normal_distribution<float> _normal;

    std::optional<DPMSolver> _solver;

    std::shared_ptr<QnnBackend> _qnn;
    std::optional<StableDiffusionModel> _model;
    std::optional<Tokenizer> _tokenizer;

    std::vector<Tokenizer::token_type> tokens_host;
    std::vector<Tokenizer::token_type> empty_prompt_host;
    std::vector<float> x_host;
    std::vector<float> y_host;
    std::vector<float> img_host;

    std::vector<std::vector<float>> t_embeddings; // sequence of encoded timesteps

    unsigned int unet_dim = 1280; // TODO: expose as arg?

    std::optional<QnnTensor> tokens;
    std::optional<QnnTensor> p_cond;
    std::optional<QnnTensor> p_uncond;
    std::optional<QnnTensor> x;
    std::optional<QnnTensor> t;
    std::optional<QnnTensor> y;
    std::optional<QnnTensor> img;

    std::list<QnnTensor> p_cond_inputs;
    std::list<QnnTensor> p_uncond_inputs;

    tensor_list other_tensors;
};

}

#endif // LIBSDOD_CONTEXT_H
