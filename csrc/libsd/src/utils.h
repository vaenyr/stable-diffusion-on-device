#ifndef LIBSD_UTILS_H
#define LIBSD_UTILS_H

#define _STR(x) #x
#define STR(x) _STR(x)

#include <cstddef>
#include <string>
#include <sstream>
#include <type_traits>
#include <functional>


namespace libsd {

namespace details {

std::size_t get_next_insertion_point(std::string const& str, std::size_t offset);
std::string format(std::size_t pos, std::string const& fmt);
std::string format(std::size_t pos, std::string&& fmt);

template <class T>
struct is_valid_std_to_string {
private:
    template <class U>
    static auto check(U&& u) -> decltype(std::to_string(std::forward<U>(u)), std::true_type{});
    static std::false_type check(...);
public:
    static constexpr auto value = decltype(check(std::declval<T&&>()))::value;
};

template <class T>
inline constexpr bool is_valid_std_to_string_v = is_valid_std_to_string<T>::value;


inline std::string to_string(std::string const& s) {
    return s;
}

inline std::string to_string(std::string&& s) {
    return std::move(s);
}

inline std::string to_string(const char* s) {
    return std::string(s);
}

template <class T>
inline std::enable_if_t<is_valid_std_to_string_v<T&&>, std::string> to_string(T&& t) {
    return std::to_string(std::forward<T>(t));
}

template <class T>
inline std::enable_if_t<std::is_pointer_v<std::remove_reference_t<T>>, std::string> to_string(T&& t) {
    std::stringstream ss;
    ss << "0x" << std::hex << reinterpret_cast<const std::uintptr_t>(t);
    return ss.str();
}


template <class Arg, class... T>
std::string format(std::size_t pos, std::string const& fmt, Arg&& arg, T&&... args) {
    pos = get_next_insertion_point(fmt, pos);
    if (pos >= fmt.length())
        return fmt;

    auto&& rep = to_string(std::forward<Arg>(arg));
    return format(pos + rep.length(), fmt.substr(0, pos) + rep + fmt.substr(pos+2), std::forward<T>(args)...);
}

template <class Arg, class... T>
std::string format(std::size_t pos, std::string&& fmt, Arg&& arg, T&&... args) {
    pos = get_next_insertion_point(fmt, pos);
    if (pos >= fmt.length())
        return std::move(fmt);

    auto&& rep = to_string(std::forward<Arg>(arg));
    return format(pos + rep.length(), fmt.substr(0, pos) + rep + fmt.substr(pos+2), std::forward<T>(args)...);
}

}

template <class... T>
std::string format(std::string const& fmt, T&&... args) {
    return details::format(0, fmt, std::forward<T>(args)...);
}

template <class... T>
std::string format(std::string&& fmt, T&&... args) {
    return details::format(0, std::move(fmt), std::forward<T>(args)...);
}

std::size_t get_file_size(std::string const& path);
bool read_file_content(std::string const& path, std::vector<unsigned char>& buffer);

}

#endif // LIBSD_UTILS_H
