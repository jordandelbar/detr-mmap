#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <cstdlib>

namespace bridge {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

enum class LogFormat {
    PRETTY,
    JSON
};

class Logger {
public:
    static void init();
    static void log(LogLevel level, const std::string& message, const std::string& target = "");

    // Helper to log with fields (simplified for manual JSON/string building)
    template<typename... Args>
    static void log_structured(LogLevel level, const std::string& message, Args... args) {
        log(level, message); // Simplified for now
    }

private:
    static LogFormat format_;
    static std::mutex mutex_;

    static std::string get_timestamp();
    static std::string level_to_string(LogLevel level);
    static std::string escape_json(const std::string& s);
};

// Global convenience macros
#define LOG_INFO(msg) bridge::Logger::log(bridge::LogLevel::INFO, msg)
#define LOG_WARN(msg) bridge::Logger::log(bridge::LogLevel::WARN, msg)
#define LOG_ERROR(msg) bridge::Logger::log(bridge::LogLevel::ERROR, msg)
#define LOG_DEBUG(msg) bridge::Logger::log(bridge::LogLevel::DEBUG, msg)

// FFI helper
void init_logger();

} // namespace bridge
