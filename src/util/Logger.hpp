// =============================================================================
// Logger.hpp
// -----------------------------------------------------------------------------
// Minimal, thread-safe logging + a CSV statistics writer.
//
//  * Logger: leveled console logging guarded by a mutex so the simulation and
//    training threads can log without interleaving characters.
//  * CsvWriter: append-only CSV used for reward curves / episode statistics so
//    results can be plotted by any external tool (gnuplot, pandas, Excel).
// =============================================================================
#pragma once

#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <string>
#include <fstream>
#include <vector>
#include <utility>

namespace dp::util {

enum class LogLevel { Trace, Debug, Info, Warn, Error };

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void setLevel(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }

    void log(LogLevel level, const char* fmt, ...) {
        if (level < level_) return;
        char buffer[1024];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        std::lock_guard<std::mutex> lock(mutex_);
        std::fprintf(level >= LogLevel::Warn ? stderr : stdout,
                     "[%s] %s\n", tag(level), buffer);
    }

private:
    static const char* tag(LogLevel level) {
        switch (level) {
            case LogLevel::Trace: return "TRACE";
            case LogLevel::Debug: return "DEBUG";
            case LogLevel::Info:  return "INFO ";
            case LogLevel::Warn:  return "WARN ";
            case LogLevel::Error: return "ERROR";
        }
        return "?????";
    }

    LogLevel   level_ = LogLevel::Info;
    std::mutex mutex_;
};

// Convenience macros (variadic). They evaluate the level check first to avoid
// formatting cost when the message would be filtered out.
#define DP_LOG_INFO(...)  ::dp::util::Logger::instance().log(::dp::util::LogLevel::Info,  __VA_ARGS__)
#define DP_LOG_WARN(...)  ::dp::util::Logger::instance().log(::dp::util::LogLevel::Warn,  __VA_ARGS__)
#define DP_LOG_ERROR(...) ::dp::util::Logger::instance().log(::dp::util::LogLevel::Error, __VA_ARGS__)
#define DP_LOG_DEBUG(...) ::dp::util::Logger::instance().log(::dp::util::LogLevel::Debug, __VA_ARGS__)

// -----------------------------------------------------------------------------
// CsvWriter: simple buffered CSV appender for training statistics.
// -----------------------------------------------------------------------------
class CsvWriter {
public:
    CsvWriter() = default;

    bool open(const std::string& path, const std::vector<std::string>& header) {
        file_.open(path, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) return false;
        for (std::size_t i = 0; i < header.size(); ++i) {
            file_ << header[i];
            if (i + 1 < header.size()) file_ << ',';
        }
        file_ << '\n';
        return true;
    }

    template <typename... Args>
    void writeRow(Args... args) {
        writeRowImpl(true, args...);
        file_ << '\n';
        file_.flush();
    }

    bool isOpen() const { return file_.is_open(); }

private:
    template <typename T, typename... Rest>
    void writeRowImpl(bool first, T value, Rest... rest) {
        if (!first) file_ << ',';
        file_ << value;
        writeRowImpl(false, rest...);
    }
    void writeRowImpl(bool /*first*/) {}

    std::ofstream file_;
};

} // namespace dp::util
