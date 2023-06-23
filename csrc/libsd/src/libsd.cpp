#include "libsd.h"

#include "errors.h"
#include "context.h"

#include <cstring>

#define LIBSD_VERSION_MAJOR 1
#define LIBSD_VERSION_MINOR 0
#define LIBSD_VERSION_PATCH 0

#define LIBSD_VERSION_STR (#LIBSD_VERSION_MAJOR "." #LIBSD_VERSION_MINOR "." #LIBSD_VERSION_PATCH)
#define LIBSD_VERSION_INT (LIBSD_VERSION_MAJOR*10000 + LIBSD_VERSION_MINOR*100 + LIBSD_VERSION_PATCH)
#define LIBSD_CONTEXT_MAGIC_HEADER 0x00534443
#define LIBSD_DEFAULT_CONTEXT_VERSION 1


namespace libsd {

struct CAPI_Context_Handler {
    unsigned int magic_info = LIBSD_CONTEXT_MAGIC_HEADER;
    unsigned int context_version = LIBSD_DEFAULT_CONTEXT_VERSION;
    unsigned int ref_count = 0;
    Context* cptr = nullptr;
};

template <class T>
ErrorCode _error(ErrorCode code, Context* c, T&& message, const char* func, const char* file, const char* line) {
    ErrorTable tab = nullptr;
    if (c)
        tab = c->get_error_table();

    auto last_sep = strrchr(file, '/');
    if (last_sep)
        file = last_sep + 1;

    std::string msg{ func };
    msg = msg + ": " + message + " [" + file + ":" + line + "]";
    record_error(tab, code, msg);
    return code;
}

#define _STR(x) #x
#define STR(x) _STR(x)
#define ERROR(code, reason) _error(code, cptr, reason, __func__, __FILE__, STR(__LINE__))


#define TRY_RETRIEVE_CONTEXT \
    Context* cptr = nullptr; \
    if (context == nullptr) \
        return ERROR(INVALID_CONTEXT, "context is nullptr"); \
    auto hnd = reinterpret_cast<CAPI_Context_Handler*>(context); \
    if (hnd->magic_info != LIBSD_CONTEXT_MAGIC_HEADER) \
        return ERROR(INVALID_CONTEXT, "context magic header mismatch! got: " + std::to_string(hnd->magic_info)); \
    if (hnd->context_version != LIBSD_DEFAULT_CONTEXT_VERSION) \
        return ERROR(INVALID_CONTEXT, "context version mismatch! got: " + std::to_string(hnd->context_version)); \
    if (hnd->ref_count == 0) \
        return ERROR(INVALID_CONTEXT, "context has been released!"); \
    if (hnd->cptr == nullptr) \
        return ERROR(INVALID_CONTEXT, "corrupted context, internal pointer is nullptr"); \
    cptr = hnd->cptr


static ErrorCode setup_impl(void** context, const char* models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, unsigned int log_level) {
    Context* cptr = nullptr;
    if (context == nullptr)
        return ERROR(INVALID_ARGUMENT, "Context argument should not be nullptr!");

    if (*context != nullptr)
        return ERROR(INVALID_ARGUMENT, "Context should point to a nullptr-initialized variable!");

    CAPI_Context_Handler* hnd = new (std::nothrow) CAPI_Context_Handler;
    if (hnd == nullptr)
        return ERROR(FAILED_ALLOCATION, "Could not create a new CAPI_Context_Handler object");

    hnd->ref_count += 1;
    hnd->cptr = new (std::nothrow) Context(models_dir, latent_channels, latent_spatial, upscale_factor, log_level);
    if (hnd->cptr == nullptr)
        return ERROR(FAILED_ALLOCATION, "COuld not create a new Context object");

    cptr = hnd->cptr;
    *context = hnd;

    try {
        cptr->initialize_qnn();
        cptr->load_models();
        cptr->prepare_sampler();
    } catch (libsd_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(INTERNAL_ERROR, "Unspecified error");
    }

    return NO_ERROR;
}

static ErrorCode ref_context_impl(void* context) {
    TRY_RETRIEVE_CONTEXT;
    ++hnd->ref_count;
    return NO_ERROR;
}

static ErrorCode release_impl(void* context) {
    TRY_RETRIEVE_CONTEXT;
    if (--hnd->ref_count == 0) {
        delete cptr;
        cptr = nullptr;
        hnd->cptr = nullptr;
    }

    return NO_ERROR;
}

static ErrorCode generate_image_impl(void* context, const char* prompt, unsigned int prompt_length, float guidance_scale, unsigned char** image_out, unsigned int* image_buffer_size) {
    TRY_RETRIEVE_CONTEXT;
    if (image_out == nullptr)
        return ERROR(INVALID_ARGUMENT, "image_out is nullptr");
    if (image_buffer_size == nullptr)
        return ERROR(INVALID_ARGUMENT, "image_buffer_size is nullptr");

    try {
        auto out = (*image_out == nullptr) ? cptr->allocate_output() : cptr->reuse_buffer(*image_out, *image_buffer_size);
        *image_out = out.data_ptr();
        *image_buffer_size = out.data_len();
        cptr->generate(std::string(prompt, prompt_length), guidance_scale, out);
    } catch (libsd_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(INTERNAL_ERROR, "Unspecified error");
    }

    return NO_ERROR;
}

static const char* ger_error_description_impl(int errorcode) {
    if (!is_valid_error_code(errorcode))
        return nullptr;

    return get_error_str(static_cast<ErrorCode>(errorcode));
}

static const char* get_last_error_extra_info_impl(int errorcode, void* context) {
    if (!is_valid_error_code(errorcode))
        return nullptr;

    ErrorTable tab = nullptr;
    if (context) {
        auto hnd = reinterpret_cast<CAPI_Context_Handler*>(context);
        if (hnd->magic_info == LIBSD_CONTEXT_MAGIC_HEADER
            && hnd->context_version == LIBSD_DEFAULT_CONTEXT_VERSION
            && hnd->ref_count > 0
            && hnd->cptr != nullptr)
            tab = hnd->cptr->get_error_table();
    }

    return get_last_error_info(tab, static_cast<ErrorCode>(errorcode));
}

}

extern "C" {

LIBSD_API int setup(void** context, const char* models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, unsigned int log_level) {
    return libsd::setup_impl(context, models_dir, latent_channels, latent_spatial, upscale_factor, log_level);
}

LIBSD_API int ref_context(void* context) {
    return libsd::ref_context_impl(context);
}

LIBSD_API int release(void* context) {
    return libsd::release_impl(context);
}

LIBSD_API int generate_image(void* context, const char* prompt, unsigned int prompt_length, float guidance_scale, unsigned char** image_out, unsigned int* image_buffer_size) {
    return libsd::generate_image_impl(context, prompt, prompt_length, guidance_scale, image_out, image_buffer_size);
}


LIBSD_API const char* get_error_description(int errorcode) {
    return libsd::get_error_description_impl(errorcode);
}

LIBSD_API const char* get_last_error_extra_info(int errorcode, void* context) {
    return libsd::get_last_error_extra_info_impl(errorcode, context);
}

}