
#ifndef LIBSDOD_API
#define LIBSDOD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif


enum libsdod_status_code {
    LIBSDOD_NO_ERROR,
    LIBSDOD_INVALID_CONTEXT,
    LIBSDOD_INVALID_ARGUMENT,
    LIBSDOD_FAILED_ALLOCATION,
    LIBSDOD_RUNTIME_ERROR,
    LIBSDOD_INTERNAL_ERROR,
};


enum libsdod_log_level {
   LIBSDOD_LOG_NOTHING,
   LIBSDOD_LOG_ERROR,
   LIBSDOD_LOG_INFO,
   LIBSDOD_LOG_DEBUG,
   LIBSDOD_LOG_ABUSIVE
};


/* Prepare models and devices to run image generation.

   context - will return prepared context (type void*) there, should not be nullptr
   models_dir - directory holding models to load, should include "decoder.bin", "cond.bin", "sd_unet_inputs.bin", "sd_unet_middle", "sd_unet_outputs.bin", "sd_unet_head.bin"
   latent_channels - latent representation channels, SD1.5 uses 4
   latent_spatial - latent representation spatial dimensions, SD1.5 uses 64
   upscale_factor - upscaling factor for the decoder, SD1.5 uses 8
   steps - number of denoising steps to perform when generating an image, can be later overwritten with libsdod_set_steps
   log_level - logging level, can be later overwritten with libsdod_set_log_level

   Returns 0 if successful, otherwise an error code is returned.
   If successful, *context will be pointer to a prepared context that should be passed to other functions and cleaned when no longer needed, see release.
   Even if unsuccessful, *context might still be set if failure happened after initial object has been created, in which case it should still be released
   by a call to ``release``, it should also be used when querying for error details; it should not be, however, used to generate images.
   If a method fails before a context object is created, *context will be nullptr.
*/
LIBSDOD_API int libsdod_setup(void** context, const char* models_dir, unsigned int latent_channels, unsigned int latent_spatial, unsigned int upscale_factor, unsigned int steps, unsigned int log_level);


/* Changes the number of denoising steps performed when generating images using the provided context.

   context - a previously prepared context obtained by a call to setup
   unsigned int steps - new number of denoising steps
   
   Returns 0 if successful, otherwise an error code is returned.
*/
LIBSDOD_API int libsdod_set_steps(void* context, unsigned int steps);


/* Changes the log level for the provided context.

   context - a previously prepared context obtained by a call to setup
   unsigned int log_level - new log level
   
   Returns 0 if successful, otherwise an error code is returned.
*/
LIBSDOD_API int libsdod_set_log_level(void* context, unsigned int log_level);


/* Increase reference counter for a given context.

   For each additional call to ref_context, an additional call to release has to be made before
   a context will be actually cleaned.
*/
LIBSDOD_API int libsdod_ref_context(void* context);

/* Release a previously prepared context, obtained by a call to setup.

   Returns 0 if successful, otherwise an error code is returned.
*/
LIBSDOD_API int libsdod_release(void* context);


/* Run a diffusion process.

   context - a previously prepared context obtained by a call to setup
   prompt - a user prompt to guide the generation process
   guidance_scale - scaling factor for the classifier-free guidance: scale * noise(x, t, p) - (1 - scale) * noise(x, t, ""), SD1.5 default is 7.5
   image_out - output buffer that will hold a resulting RGB image, in [H, W, C] order, values in range 0-255

   This function can either handle memory allocation on its own or work with a user-provided buffer.
   If the user wants to leave allocation to the function, please call it as:

    unsigned char* buffer = nullptr;
    unsigned int buffer_len = 0;

    generate_image(..., &buffer, &buffer_len);

   In which case the function will write to both ``buffer`` and ``buffer_len``. Note that setting initial value of ``buffer`` to ``nullptr``
   is important! Otherwise the code will assume a user-provided buffer with length 0 is to be used, which will result in an error.
   Otherwise, if user wants to reuse an existing buffer, the call should be made as:

    unsigned char* buffer = <existing_buffer>;
    unsigned int buffer_len = <length of the existing buffer>;

    generate_image(..., &buffer, &buffer_len);

   If the existing buffer is too small, the function will return an appropriate error.
   If it is too large, the function will continue its work as normal, and will return back the amount of data actually written to the buffer
   using the same ``buffer_len`` variable (similar to the case when allocation is performed by the function). Therefore it is important
   to keep the length of the original buffer as a separate variable - otherwise this information might be lost.

   In either case (allocation handled by the user, or by the function), it is the user's responsibility to free the buffer when it is no longer needed.

   Returns 0 if successful, otherwise an error code is returned.
*/
LIBSDOD_API int libsdod_generate_image(void* context, const char* prompt, float guidance_scale, unsigned char** image_out, unsigned int* image_buffer_size);


/* Return a human-readable null-terminated string describing a returned error code.

   The method can return nullptr if ``errorcode`` is not a valid error code.
*/
LIBSDOD_API const char* libsdod_get_error_description(int errorcode);


/* Return extra information about the last error with ``errorcode`` that occurred within a given ``context``.

   In general, the information about each error is stored per-error and per-context. There are two exceptions, though,
   when it is only stored per-error:
      1. If an error occurred while setting up a context, the user should pass nullptr as ``context`` to obtain necessary information.
      2. If an error is related to calling a function with invalid context (indicated by certain error codes) then ``context`` is also ignored.

   The method can return nullptr if either ``errorcode`` is not a valid error code, if ``context`` is not a valid context,
   if ``errorcode`` has not happened for ``context`` or if no extra information has been provided by the implementation when an error was recorded.
   Otherwise a null-terminated string is returned.
*/
LIBSDOD_API const char* libsdod_get_last_error_extra_info(int errorcode, void* context);

#ifdef __cplusplus
}
#endif
