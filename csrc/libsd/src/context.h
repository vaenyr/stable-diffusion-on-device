#ifndef LIBSD_CONTEXT_H
#define LIBSD_CONTEXT_H

#include <string>

namespace libsd {

class Context {
public:
    Context(std::string const& models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, unsigned int log_level);
    virtual ~Context();

    void initialize_qnn();
    void load_models();
    void prepare_sampler();

    void generate(std::string const& prompt, float guidance, Buffer<unsigned char> output);

private:
    std::string models_dir;
    unsigned int latent_channels;
    unsigned int latent_spatial;
    unsigned int upscale_factor;
    unsigned int log_level;

    bool _failed_and_gave_up = false;
    bool _qnn_initialized = false;
    bool _models_loaded = false;

    void* _error_table = nullptr;

    void _init_qnn();
};

}

#endif // LIBSD_CONTEXT_H
