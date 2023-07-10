#include "libsdod.h"

#include "errors.h"
#include "context.h"
#include "utils.h"

#include <string>
#include <cstring>

#define LIBSDOD_VERSION_MAJOR 1
#define LIBSDOD_VERSION_MINOR 0
#define LIBSDOD_VERSION_PATCH 0

#define LIBSDOD_VERSION_STR (#LIBSDOD_VERSION_MAJOR "." #LIBSDOD_VERSION_MINOR "." #LIBSDOD_VERSION_PATCH)
#define LIBSDOD_VERSION_INT (LIBSDOD_VERSION_MAJOR*10000 + LIBSDOD_VERSION_MINOR*100 + LIBSDOD_VERSION_PATCH)
#define LIBSDOD_CONTEXT_MAGIC_HEADER 0x00534443
#define LIBSDOD_DEFAULT_CONTEXT_VERSION 1


namespace libsdod {

struct CAPI_Context_Handler {
    unsigned int magic_info = LIBSDOD_CONTEXT_MAGIC_HEADER;
    unsigned int context_version = LIBSDOD_DEFAULT_CONTEXT_VERSION;
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

#define ERROR(code, reason) _error(code, cptr, reason, __func__, __FILE__, STR(__LINE__))


#define TRY_RETRIEVE_CONTEXT \
    Context* cptr = nullptr; \
    if (context == nullptr) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context is nullptr"); \
    auto hnd = reinterpret_cast<CAPI_Context_Handler*>(context); \
    if (hnd->magic_info != LIBSDOD_CONTEXT_MAGIC_HEADER) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context magic header mismatch! got: " + std::to_string(hnd->magic_info)); \
    if (hnd->context_version != LIBSDOD_DEFAULT_CONTEXT_VERSION) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context version mismatch! got: " + std::to_string(hnd->context_version)); \
    if (hnd->ref_count == 0) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context has been released!"); \
    if (hnd->cptr == nullptr) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "corrupted context, internal pointer is nullptr"); \
    cptr = hnd->cptr; \
    auto&& _logger_scope = cptr->activate_logger(); \
    (void)_logger_scope


static ErrorCode setup_impl(void** context, const char* models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, unsigned int steps, unsigned int log_level) {
    Context* cptr = nullptr;
    if (context == nullptr)
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Context argument should not be nullptr!");

    if (*context != nullptr)
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Context should point to a nullptr-initialized variable!");

    if (!is_valid_log_level(log_level))
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid log_level");

    CAPI_Context_Handler* hnd = new (std::nothrow) CAPI_Context_Handler;
    if (hnd == nullptr)
        return ERROR(ErrorCode::FAILED_ALLOCATION, "Could not create a new CAPI_Context_Handler object");

    hnd->ref_count += 1;
    hnd->cptr = new (std::nothrow) Context(models_dir, latent_channels, latent_spatial, upscale_factor, static_cast<LogLevel>(log_level));
    if (hnd->cptr == nullptr)
        return ERROR(ErrorCode::FAILED_ALLOCATION, "COuld not create a new Context object");

    cptr = hnd->cptr;
    *context = hnd;
    auto&& _logger_scope = cptr->activate_logger();
    (void)_logger_scope;

    try {
#ifndef NOTHREADS
        cptr->init_mt(steps);
#else
        cptr->initialize_qnn();
        cptr->load_models();
        cptr->load_tokenizer();
        cptr->prepare_buffers();
        cptr->prepare_solver();
        cptr->prepare_schedule(steps);
#endif
    } catch (libsdod_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(ErrorCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(ErrorCode::INTERNAL_ERROR, "Unspecified error");
    }

    return ErrorCode::NO_ERROR;
}

static ErrorCode set_steps_impl(void* context, unsigned int steps) {
    TRY_RETRIEVE_CONTEXT;
    try {
        cptr->prepare_schedule(steps);
    } catch (libsdod_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(ErrorCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(ErrorCode::INTERNAL_ERROR, "Unspecified error");
    }

    return ErrorCode::NO_ERROR;
}

static ErrorCode set_log_level_impl(void* context, unsigned int log_level) {
    TRY_RETRIEVE_CONTEXT;
    if (!is_valid_log_level(log_level))
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid log_level");

    try {
        cptr->get_logger().set_level(static_cast<LogLevel>(log_level));
    } catch (libsdod_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(ErrorCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(ErrorCode::INTERNAL_ERROR, "Unspecified error");
    }

    return ErrorCode::NO_ERROR;
}

static ErrorCode ref_context_impl(void* context) {
    TRY_RETRIEVE_CONTEXT;
    ++hnd->ref_count;
    return ErrorCode::NO_ERROR;
}

static ErrorCode release_impl(void* context) {
    TRY_RETRIEVE_CONTEXT;
    if (--hnd->ref_count == 0) {
        delete cptr;
        cptr = nullptr;
        hnd->cptr = nullptr;
    }

    return ErrorCode::NO_ERROR;
}

static ErrorCode generate_image_impl(void* context, const char* prompt, float guidance_scale, unsigned char** image_out, unsigned int* image_buffer_size) {
    TRY_RETRIEVE_CONTEXT;
    if (image_out == nullptr)
        return ERROR(ErrorCode::INVALID_ARGUMENT, "image_out is nullptr");
    if (image_buffer_size == nullptr)
        return ERROR(ErrorCode::INVALID_ARGUMENT, "image_buffer_size is nullptr");

    try {
        auto out = (*image_out == nullptr) ? cptr->allocate_output() : cptr->reuse_buffer(*image_out, *image_buffer_size);
        cptr->generate(std::string(prompt), guidance_scale, out);
        *image_out = out.data_ptr();
        *image_buffer_size = out.data_len();
        out.own(false);
    } catch (libsdod_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(ErrorCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(ErrorCode::INTERNAL_ERROR, "Unspecified error");
    }

    return ErrorCode::NO_ERROR;
}

static const char* get_error_description_impl(int errorcode) {
    if (!is_valid_error_code(errorcode))
        return nullptr;

    return get_error_str(static_cast<ErrorCode>(errorcode));
}

static const char* get_last_error_extra_info_impl(int errorcode, void* context) {
    if (!is_valid_error_code(errorcode))
        return nullptr;

    ErrorTable tab = nullptr;
    if (context && errorcode != std::underlying_type_t<ErrorCode>(ErrorCode::INVALID_CONTEXT)) {
        auto hnd = reinterpret_cast<CAPI_Context_Handler*>(context);
        if (hnd->magic_info == LIBSDOD_CONTEXT_MAGIC_HEADER
            && hnd->context_version == LIBSDOD_DEFAULT_CONTEXT_VERSION
            && hnd->ref_count > 0
            && hnd->cptr != nullptr)
            tab = hnd->cptr->get_error_table();
    }

    return get_last_error_info(tab, static_cast<ErrorCode>(errorcode));
}

}

extern "C" {

LIBSDOD_API int libsdod_setup(void** context, const char* models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, unsigned int steps, unsigned int log_level) {
    return static_cast<int>(libsdod::setup_impl(context, models_dir, latent_channels, latent_spatial, upscale_factor, steps, log_level));
}

LIBSDOD_API int libsdod_set_steps(void* context, unsigned int steps) {
    return static_cast<int>(libsdod::set_steps_impl(context, steps));
}

LIBSDOD_API int libsdod_set_log_level(void* context, unsigned int log_level) {
    return static_cast<int>(libsdod::set_log_level_impl(context, log_level));
}

LIBSDOD_API int libsdod_ref_context(void* context) {
    return static_cast<int>(libsdod::ref_context_impl(context));
}

LIBSDOD_API int libsdod_release(void* context) {
    return static_cast<int>(libsdod::release_impl(context));
}

LIBSDOD_API int libsdod_generate_image(void* context, const char* prompt, float guidance_scale, unsigned char** image_out, unsigned int* image_buffer_size) {
    return static_cast<int>(libsdod::generate_image_impl(context, prompt, guidance_scale, image_out, image_buffer_size));
}

LIBSDOD_API const char* libsdod_get_error_description(int errorcode) {
    return libsdod::get_error_description_impl(errorcode);
}

LIBSDOD_API const char* libsdod_get_last_error_extra_info(int errorcode, void* context) {
    return libsdod::get_last_error_extra_info_impl(errorcode, context);
}

}
