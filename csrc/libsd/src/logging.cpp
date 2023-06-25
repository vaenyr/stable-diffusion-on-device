#include "logging.h"

#include <iostream>
#include <utility>

#ifdef __ANDROID__
#include <android/log.h>
#else
// dummy values, since they won't be used anyway
#define ANDROID_LOG_DEBUG 0
#define ANDROID_LOG_INFO 0
#define ANDROID_LOG_ERROR 0
#endif

namespace libsd {

namespace {

thread_local Logger* active_logger = nullptr;

void dispatch_message(std::ostream& out, uint64_t timestamp, const char* level_tag, int android_prio, const char* message) {
    out << "[" << timestamp << "]:" << level_tag << " " << message << std::endl;
#ifdef __ANDROID__
    __android_log_write(android_prio, "[LibSD]", message);
#endif
}

} // end local

bool is_valid_log_level(int log_level) {
    return log_level >= 0 && log_level < LIBSD_NUM_LOG_LEVELS;
}

bool is_enabled(LogLevel level) {
    if (!active_logger)
        return false;
    return active_logger->get_level() <= level;
}

void message(LogLevel level, std::string const& str) {
    if (active_logger)
        active_logger->message(level, str);
}

void message(uint64_t timestamp, LogLevel level, std::string const& str) {
    if (active_logger)
        active_logger->message(timestamp, level, str);
}

} // end libsd functions

using namespace libsd;

Logger::Logger() : current_level(LogLevel::NOTHING) {
}

Logger::~Logger() {
    if (active_logger == this)
        active_logger = nullptr;
}

void Logger::set_level(LogLevel level) {
    current_level = level;
}

void Logger::message(uint64_t timestamp, LogLevel level, std::string const& str) {
    if (current_level > level || current_level >= LogLevel::NOTHING)
        return;

    switch (level) {
    case LogLevel::DEBUG:
        dispatch_message(std::cout, timestamp, "[DEBUG]", ANDROID_LOG_DEBUG, str.c_str());
        break;
    case LogLevel::INFO:
        dispatch_message(std::cout, timestamp, "[INFO]", ANDROID_LOG_INFO, str.c_str());
        break;
    case LogLevel::ERROR:
        dispatch_message(std::cerr, timestamp, "[ERROR]", ANDROID_LOG_ERROR, str.c_str());
        break;
    case LogLevel::NOTHING:
        std::unreachable();
    }
}

ActiveLoggerScopeGuard::ActiveLoggerScopeGuard(Logger& logger) : prev(active_logger) {
    active_logger = &logger;
}

ActiveLoggerScopeGuard::ActiveLoggerScopeGuard(ActiveLoggerScopeGuard&& other) : prev(other.prev) {
    other.active = false;
}

ActiveLoggerScopeGuard::~ActiveLoggerScopeGuard() {
    if (active)
        active_logger = prev;
}
