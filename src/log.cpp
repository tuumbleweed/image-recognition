#include "log.h"

// use info by default
SimpleLogger::LogLevel SimpleLogger::global_log_level_ = SimpleLogger::INFO;
// for thread safety
std::mutex SimpleLogger::log_mutex_;

SimpleLogger::SimpleLogger(LogLevel level) : log_level_(level) {}

SimpleLogger::~SimpleLogger() {
    std::lock_guard<std::mutex> guard(log_mutex_);
    std::cerr << oss_.str();
    if (log_level_ == FATAL) {
        std::cerr << "Program terminated due to fatal error." << std::endl;
        std::terminate(); // Terminate the program on fatal error
    }
}

void SimpleLogger::set_log_level(LogLevel level) {
    global_log_level_ = level;
}

bool SimpleLogger::should_log(LogLevel level) {
    return level <= global_log_level_;
}

std::string SimpleLogger::trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos)
        return str;
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
}


void SimpleLogger::load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Could not open config file: " << config_file << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));

            if (key == "log_level") {
                set_log_level(string_to_log_level(value));
            }
        }
    }
}

std::string SimpleLogger::log_level_to_string(LogLevel level) {
    switch (level) {
        case FATAL: return "FATAL";
        case ERROR: return "ERROR";
        case INFO:  return "INFO";
        case DEBUG: return "DEBUG";
        case TRACE: return "TRACE";
        default:    return "UNKNOWN";
    }
}

SimpleLogger::LogLevel SimpleLogger::string_to_log_level(const std::string& level_str) {
    if (level_str == "FATAL") return FATAL;
    if (level_str == "ERROR") return ERROR;
    if (level_str == "INFO") return INFO;
    if (level_str == "DEBUG") return DEBUG;
    if (level_str == "TRACE") return TRACE;
    return INFO; // Default to INFO if unknown
}
