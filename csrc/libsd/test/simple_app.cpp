#include <iostream>

#include "libsd.h"

int main() {
    void* ctx = nullptr;
    int status = libsd_setup(&ctx, "../../../../dlc", 4, 64, 8, 0);
    if (status) {
        std::cout << "Initialization error: " << libsd_get_error_description(status) << "; " << libsd_get_last_error_extra_info(status, ctx) << std::endl;
        if (ctx)
            libsd_release(ctx);
        return 1;
    }

    unsigned char* img = nullptr;
    unsigned int img_len = 0;

    status = libsd_generate_image(ctx, "A photograph of an astronaut riding a horse", 7.5f, &img, &img_len);
    if (status) {
        std::cout << "Generation error: " << libsd_get_error_description(status) << "; " << libsd_get_last_error_extra_info(status, ctx) << std::endl;
        if (ctx)
            libsd_release(ctx);
        return 1;
    }

    delete[] img;
    libsd_release(ctx);
    return 0;
}
