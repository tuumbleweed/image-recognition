#ifndef SIMPLE_LOGGER_H
#define SIMPLE_LOGGER_H

#include <iostream>
#include <sstream>
#include <string>
#include <mutex>
#include <fstream>
#include <map>

class SimpleLogger {
public:
    enum LogLevel {
        FATAL = 0,
        ERROR = 1,
        INFO = 2,
        DEBUG = 3,
        TRACE = 4
    };

    SimpleLogger(LogLevel level);
    ~SimpleLogger();

    template <typename T>
    SimpleLogger& operator<<(const T& value) {
        oss_ << value;
        return *this;
    }

    // Overload for handling std::endl and other manipulators
    SimpleLogger& operator<<(std::ostream& (*manip)(std::ostream&)) {
        oss_ << manip;
        return *this;
    }

    static void set_log_level(LogLevel level);
    static bool should_log(LogLevel level);
    static void load_config(const std::string& config_file);

private:
    LogLevel log_level_;
    std::ostringstream oss_;
    static LogLevel global_log_level_;
    static std::mutex log_mutex_;

    static std::string log_level_to_string(LogLevel level);
    static LogLevel string_to_log_level(const std::string& level_str);
    static std::string trim(const std::string& str);
};

#define LOG(level) if (SimpleLogger::should_log(level)) SimpleLogger(level)

#endif // SIMPLE_LOGGER_H
