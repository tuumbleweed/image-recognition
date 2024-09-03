#include "io/log.h"

#include "io/colors.h"

// use info by default
SL::LogLevel SL::global_log_level_ = SL::INFO;
// for thread safety
std::mutex SL::log_mutex_;

SL::SL(LogLevel level) : log_level_(level) {}

SL::~SL() {
    // The mutex will remain locked for the duration of the guard object's lifetime.
    // Once the guard object goes out of scope (typically at the end of the
    // block in which it is declared), its destructor is called, which automatically unlocks the mutex.
    std::lock_guard<std::mutex> guard(log_mutex_);
    switch (log_level_) {
        case FATAL:
            std::cerr << BG_RED << oss_.str() << RESET;
            break;
        case ERROR:
            std::cerr << RED << oss_.str() << RESET;
            break;
        case WARNING:
            std::cerr << YELLOW << oss_.str() << RESET;
            break;
        default:
            std::cerr << oss_.str();
            break;
    }
    if (log_level_ == FATAL) {
        std::cerr << DIM_RED << "\nProgram terminated due to fatal error." << std::endl;
        std::terminate();  // Terminate the program on fatal error
        std::cerr << RESET;
    }
}

void SL::set_log_level(LogLevel level) {
    std::cerr << "Setting log level to " << BLUE << log_level_to_string(level) << RESET << std::endl;
    global_log_level_ = level;
}

bool SL::should_log(LogLevel level) { return level <= global_log_level_; }

std::string SL::trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return str;
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
}

void SL::load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Could not open config file: " << config_file << std::endl;
        return;
    }

    // read key and value split by '='
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

std::string SL::log_level_to_string(LogLevel level) {
    switch (level) {
        case FATAL:
            return "FATAL";
        case ERROR:
            return "ERROR";
        case WARNING:
            return "WARNING";
        case NOTICE:
            return "NOTICE";
        case INFO:
            return "INFO";
        case DEBUG:
            return "DEBUG";
        case TRACE:
            return "TRACE";
        default:
            return "UNKNOWN";
    }
}

SL::LogLevel SL::string_to_log_level(const std::string& level_str) {
    std::cerr << "-l is " << BLUE << level_str << RESET << std::endl;
    if (level_str == "FATAL") return FATAL;
    if (level_str == "ERROR") return ERROR;
    if (level_str == "WARNING") return WARNING;
    if (level_str == "NOTICE") return NOTICE;
    if (level_str == "INFO") return INFO;
    if (level_str == "DEBUG") return DEBUG;
    if (level_str == "TRACE") return TRACE;
    return INFO;  // Default to INFO if unknown
}
