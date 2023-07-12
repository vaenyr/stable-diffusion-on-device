#include <iostream>
#include <fstream>

#include "libsdod.h"

int main() {
    void* ctx = nullptr;
#ifdef __ANDROID__
    int status = libsdod_setup(&ctx, "/data/local/tmp/libsdod", 4, 64, 8, 20, LIBSDOD_LOG_INFO, 1);
#else
    int status = libsdod_setup(&ctx, "../../../../dlc", 4, 64, 8, 20, LIBSDOD_LOG_DEBUG, 1);
#endif
    if (status) {
        std::cout << "Initialization error: " << libsdod_get_error_description(status) << "; " << libsdod_get_last_error_extra_info(status, ctx) << std::endl;
        if (ctx)
            libsdod_release(ctx);
        return 1;
    }

    unsigned char* img = nullptr;
    unsigned int img_len = 0;

    status = libsdod_generate_image(ctx, "A photograph of an astronaut riding a horse", 7.5f, &img, &img_len);
    if (status) {
        std::cout << "Generation error: " << libsdod_get_error_description(status) << "; " << libsdod_get_last_error_extra_info(status, ctx) << std::endl;
        if (ctx)
            libsdod_release(ctx);
        return 1;
    }

    std::ofstream out{ "output.bin", std::ios::binary };
    out.write(reinterpret_cast<const char*>(img), img_len);
    out.close();

    delete[] img;
    libsdod_release(ctx);
    return 0;
}
