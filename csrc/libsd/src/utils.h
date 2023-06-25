#ifndef LIBSD_UTILS_H
#define LIBSD_UTILS_H

#define _STR(x) #x
#define STR(x) _STR(x)

namespace libsd {

namespace details {

std::size_t get_next_insertion_point(std::string const& str, std::size_t offset);
std::string format(std::size_t pos, std::string const& fmt);
std::string format(std::size_t pos, std::string&& fmt);

template <class Arg, class... T>
std::string format(std::size_t pos, std::string const& fmt, Arg&& arg, T&&... args) {
    pos = get_next_insertion_point(fmt, pos);
    if (pos >= fmt.length())
        return fmt;

    auto&& rep = std::to_string(std::forward<Arg>(arg));
    return format(pos + rep.length() + 1, fmt.substr(0, pos) + rep + fmt.substr(pos+1), std::forward<T>(args)...);
}

template <class Arg, class... T>
std::string format(std::size_t pos, std::string&& fmt, Arg&& arg, T&&... args) {
    pos = get_next_insertion_point(fmt, pos);
    if (pos >= fmt.length())
        return std::move(fmt);

    auto&& rep = std::to_string(std::forward<Arg>(arg));
    return format(pos + rep.length() + 1, fmt.substr(0, pos) + rep + fmt.substr(pos+1), std::forward<T>(args)...);
}

}

template <class... T>
std::string format(std::string const& fmt, T&&... args) {
    return details::format(fmt, std::forward<T>(args)...);
}

template <class... T>
std::string format(std::string&& fmt, T&&... args) {
    return details::format(std::move(fmt), std::forward<T>(args)...);
}

}

#endif // LIBSD_UTILS_H
