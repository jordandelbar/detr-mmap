#include "logging.hpp"
#include <algorithm>

namespace bridge {

LogFormat Logger::format_ = LogFormat::PRETTY;
std::mutex Logger::mutex_;

void Logger::init() {
    const char* env_p = std::getenv("ENVIRONMENT");
    if (env_p && std::string(env_p) == "production") {
        format_ = LogFormat::JSON;
    } else {
        format_ = LogFormat::PRETTY;
    }
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    oss << "Z";
    return oss.str();
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

std::string Logger::escape_json(const std::string& s) {
    std::ostringstream o;
    for (auto c = s.cbegin(); c != s.cend(); c++) {
        switch (*c) {
        case '"': o << "\\\""; break;
        case '\\': o << "\\\\"; break;
        case '\b': o << "\\b"; break;
        case '\f': o << "\\f"; break;
        case '\n': o << "\\n"; break;
        case '\r': o << "\\r"; break;
        case '\t': o << "\\t"; break;
        default:
            if ('\x00' <= *c && *c <= '\x1f') {
                o << "\\u"
                  << std::hex << std::setw(4) << std::setfill('0') << (int)*c;
            } else {
                o << *c;
            }
        }
    }
    return o.str();
}

void Logger::log(LogLevel level, const std::string& message, const std::string& target) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostream& stream = (level == LogLevel::ERROR || level == LogLevel::WARN) ? std::cerr : std::cout;

    if (format_ == LogFormat::JSON) {
        stream << "{"
               << "\"timestamp\":\"" << get_timestamp() << "\","
               << "\"level\":\"" << level_to_string(level) << "\","
               << "\"message\":\"" << escape_json(message) << """";

        if (!target.empty()) {
            stream << ",\"target\":\"" << escape_json(target) << "\""
;        }

        stream << "}" << std::endl;
    } else {
        // Pretty format: [TIMESTAMP] [LEVEL] Message
        // Use ANSI colors for levels
        std::string color_code;
        switch (level) {
            case LogLevel::DEBUG: color_code = "\033[34m"; break; // Blue
            case LogLevel::INFO:  color_code = "\033[32m"; break; // Green
            case LogLevel::WARN:  color_code = "\033[33m"; break; // Yellow
            case LogLevel::ERROR: color_code = "\033[31m"; break; // Red
        }
        const std::string reset_code = "\033[0m";

        stream << get_timestamp() << " "
               << color_code << level_to_string(level) << reset_code << " "
               << message << std::endl;
    }
}

void init_logger() {
    Logger::init();
}

} // namespace bridge
