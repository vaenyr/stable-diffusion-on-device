#include "utils.h"

#include <string>
#include <fstream>
#include <cstddef>
#include <vector>


namespace libsdod {

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


std::size_t get_file_size(std::string const& path) {
    std::ifstream in(path, std::ifstream::binary);
    if (!in)
        return 0;

    in.seekg(0, in.end);
    auto&& length = in.tellg();
    in.seekg(0, in.beg);
    return length;
}

bool read_file_content(std::string const& path, std::vector<unsigned char>& buffer) {
    std::ifstream in(path, std::ifstream::binary);
    if (!in)
        return false;

    in.seekg(0, in.end);
    auto&& length = in.tellg();
    in.seekg(0, in.beg);

    buffer.resize(length);
    if (buffer.size() != length)
        return false;

    if (!in.read(reinterpret_cast<char*>(buffer.data()), buffer.size()))
        return false;

    return true;
}


scope_guard::scope_guard(std::function<void()> const& init, std::function<void()> deinit) {
    if (init)
        init();
    this->deinit.swap(deinit);
}


scope_guard::scope_guard(std::function<void()> deinit) : deinit(std::move(deinit)) {
}


scope_guard::scope_guard(scope_guard&& other) : deinit(std::move(other.deinit)) {

}


scope_guard::~scope_guard() {
    if (deinit)
        deinit();
}


}
