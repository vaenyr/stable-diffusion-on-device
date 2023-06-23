#include "errors.h"

namespace libsd {

static const char* _error_messages[LIBSD_NUM_ERRORS] = {
    "No error",
    "Invalid context",
    "Invalid argument",
    "Failed to allocate memory or initialise an object",
    "Runtime error occurred",
    "Internal error occurred"
};


ErrorTable allocate_error_table() {
    auto ret = new const char*[LIBSD_NUM_ERRORS];
    for (int i=0; i<LIBSD_NUM_ERRORS; ++i)
        ret[i] = nullptr;
    return ret;
}


static const char** _contextless_error_table[LIBSD_NUM_ERRORS] = { nullptr };

}