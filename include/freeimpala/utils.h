// utils.h
#pragma once

#include <memory> // For std::shared_ptr
#include <string>
#include <stdexcept>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h" // Required for stderr_color_sink_mt

namespace Utils {

// Convert string log level to spdlog level enum
spdlog::level::level_enum string_to_log_level(const std::string& level_str) {
    if (level_str == "trace") return spdlog::level::trace;
    if (level_str == "debug") return spdlog::level::debug;
    if (level_str == "info") return spdlog::level::info;
    if (level_str == "warn") return spdlog::level::warn;
    if (level_str == "error") return spdlog::level::err;
    if (level_str == "critical") return spdlog::level::critical;
    if (level_str == "off") return spdlog::level::off;
    
    throw std::invalid_argument("Invalid log level: " + level_str);
}

void init_logs(const std::string& log_level = "info") {
    // Create a colored stderr sink. This sink directs all log messages
    // to the standard error stream (stderr), often displayed in the terminal.
    // The '_mt' suffix indicates it's multi-thread safe.
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();

    // Create a new logger instance. We give it a name, "console_logger",
    // and tell it to use our stderr_sink.
    auto logger = std::make_shared<spdlog::logger>("console_logger", stderr_sink);

    // Customize the log pattern to remove the logger name ([console_logger])
    // Default pattern: "[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v"
    // Custom pattern: "[%Y-%m-%d %H:%M:%S.%e] [%l] %v" (removed %n)
    logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

    // Set this newly created logger as the default logger for spdlog.
    // This means that any calls to global spdlog functions like
    // spdlog::info(), spdlog::warn(), etc., will use this logger
    // and thus output to stderr.
    spdlog::set_default_logger(logger);

    // Convert string log level to spdlog enum and set it
    try {
        auto level = string_to_log_level(log_level);
        spdlog::set_level(level);
    } catch (const std::invalid_argument& e) {
        // If invalid level provided, default to info and log a warning
        spdlog::set_level(spdlog::level::info);
        spdlog::warn("Invalid log level '{}', defaulting to 'info'", log_level);
    }
}

} // namespace Utils