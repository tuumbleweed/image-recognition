#ifndef SIMPLE_LOGGER_H
#define SIMPLE_LOGGER_H

#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

class SL {
   public:
    enum LogLevel { FATAL = 0, ERROR = 1, WARNING = 2, NOTICE = 3, INFO = 4, DEBUG = 5, TRACE = 6 };

    SL(LogLevel level);
    ~SL();

    template <typename T>
    SL& operator<<(const T& value) {
        oss_ << value;
        return *this;
    }

    // Overload for handling std::endl and other manipulators
    SL& operator<<(std::ostream& (*manip)(std::ostream&)) {
        oss_ << manip;
        return *this;
    }

    static void set_log_level(LogLevel level);
    static bool should_log(LogLevel level);
    static void load_config(const std::string& config_file);
    static LogLevel string_to_log_level(const std::string& level_str);

   private:
    LogLevel log_level_;
    std::ostringstream oss_;
    static LogLevel global_log_level_;
    static std::mutex log_mutex_;

    static std::string log_level_to_string(LogLevel level);
    static std::string trim(const std::string& str);
};

#define LOG(level) \
    if (SL::should_log(level)) SL(level)

#endif  // SIMPLE_LOGGER_H
