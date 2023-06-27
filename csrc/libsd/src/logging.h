#ifndef LIBSD_LOGGING_H
#define LIBSD_LOGGING_H

#include <string>
#include <ctime>

#include "utils.h"


namespace libsd {

enum class LogLevel : int {
    DEBUG,
    INFO,
    ERROR,
    NOTHING
};

constexpr unsigned int LIBSD_NUM_LOG_LEVELS = 4;

bool is_valid_log_level(int loglevel);
bool is_enabled(LogLevel level);
void set_level(LogLevel level);
void message(LogLevel level, std::string const& str);
void message(uint64_t timestamp, LogLevel level, std::string const& str);

template <class... T>
void info(std::string const& fmt, T&&... args) {
    if (!is_enabled(LogLevel::INFO))
        return;
    return message(LogLevel::INFO, format(fmt, std::forward<T>(args)...));
}

template <class... T>
void debug(std::string const& fmt, T&&... args) {
    if (!is_enabled(LogLevel::DEBUG))
        return;
    return message(LogLevel::DEBUG, format(fmt, std::forward<T>(args)...));
}

template <class... T>
void error(std::string const& fmt, T&&... args) {
    if (!is_enabled(LogLevel::ERROR))
        return;
    return message(LogLevel::ERROR, format(fmt, std::forward<T>(args)...));
}


class Logger {
public:
    Logger();
    virtual ~Logger();

    void set_level(LogLevel level);
    LogLevel get_level() const { return current_level; }

    void message(LogLevel level, std::string const& str) { return message(std::time(nullptr), level, str); }
    void message(uint64_t timestamp, LogLevel level, std::string const& str);

private:
    LogLevel current_level;
    uint64_t created;
};


class ActiveLoggerScopeGuard {
public:
    ActiveLoggerScopeGuard(Logger& logger);
    ActiveLoggerScopeGuard(ActiveLoggerScopeGuard&& other);
    ActiveLoggerScopeGuard(ActiveLoggerScopeGuard const&) = delete;
    ~ActiveLoggerScopeGuard();

    ActiveLoggerScopeGuard& operator =(ActiveLoggerScopeGuard const&) = delete;
    ActiveLoggerScopeGuard& operator =(ActiveLoggerScopeGuard&&) = delete;

private:
    bool active = true;
    Logger* prev = nullptr;
};


}

#endif // LIBSD_LOGGING_H
