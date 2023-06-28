#ifndef LIBSD_UTILS_H
#define LIBSD_UTILS_H

#define _STR(x) #x
#define STR(x) _STR(x)

#include <cstddef>
#include <string>
#include <sstream>
#include <type_traits>
#include <functional>
#include <vector>
#include <list>
#include <map>
#include <span>


namespace libsd {

template <class T>
inline std::string hex(T&& t) {
    std::stringstream ss;
    ss << "0x" << std::hex << t;
    return ss.str();
}

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
    return hex(reinterpret_cast<const std::uintptr_t>(t));
}

template <class T>
inline std::string to_string(std::vector<T> const& v) {
    std::string ret = "[";
    bool first = true;
    for (auto&& e : v) {
        if (first)
            first = false;
        else
            ret.append(", ");
        ret.append(to_string(e));
    }
    ret.append("]");
    return ret;
}

template <class T>
inline std::string to_string(std::list<T> const& v) {
    std::string ret = "[";
    bool first = true;
    for (auto&& e : v) {
        if (first)
            first = false;
        else
            ret.append(", ");
        ret.append(to_string(e));
    }
    ret.append("]");
    return ret;
}

template <class T, std::size_t Extend>
inline std::string to_string(std::span<T, Extend> const& v) {
    std::string ret = "[";
    bool first = true;
    for (auto&& e : v) {
        if (first)
            first = false;
        else
            ret.append(", ");
        ret.append(to_string(e));
    }
    ret.append("]");
    return ret;
}

template <class K, class V>
inline std::string to_string(std::map<K, V> const& v) {
    std::string ret = "{";
    bool first = true;
    for (auto&& e : v) {
        if (first)
            first = false;
        else
            ret.append(", ");
        ret.append(to_string(e.first));
        ret.append(": ");
        ret.append(to_string(e.second));
    }
    ret.append("}");
    return ret;
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

template <class T>
struct range_t {
    range_t& begin() { return *this; }
    range_t& end() { return *this; }
    range_t& operator ++() { ++min; return *this; }
    T& operator *() { return min; }
    bool operator==(range_t const& r) const { return min == r.max; }

    T min;
    T max;
};

} //details

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


template <class T>
auto range(T&& max) {
    return details::range_t<std::remove_reference_t<T>>{ .min=std::remove_reference_t<T>(), .max=std::forward<T>(max) };
}


template <class T>
auto range(T&& min, T&& max) {
    return details::range_t<std::remove_reference_t<T>>{ .min=std::forward<T>(min), .max=std::forward<T>(max) };
}

}

#endif // LIBSD_UTILS_H
