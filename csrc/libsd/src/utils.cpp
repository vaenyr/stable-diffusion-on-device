#include "format.h"

namespace libsd {

namespace details {

std::size_t get_next_insertion_point(std::string const& str, std::size_t offset) {
    for (std::size_t i=offset; i<str.length()-1; ++i) {
        if (str[i] == '{' && str[i+1] == '}') {
            return i;
        }
    }

    return str.length();
}

std::string format(std::size_t pos, std::string const& fmt) {
    return fmt;
}

std::string format(std::size_t pos, std::string&& fmt) {
    return std::move(fmt);
}

}

}
