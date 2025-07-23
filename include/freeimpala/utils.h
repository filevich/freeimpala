// utils.h
#pragma once

#include <memory> // For std::shared_ptr
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h" // Required for stderr_color_sink_mt

namespace Utils {

void init_logs() {
    // Create a colored stderr sink. This sink directs all log messages
    // to the standard error stream (stderr), often displayed in the terminal.
    // The '_mt' suffix indicates it's multi-thread safe.
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();

    // Create a new logger instance. We give it a name, "console_logger",
    // and tell it to use our stderr_sink.
    auto logger = std::make_shared<spdlog::logger>("console_logger", stderr_sink);

    // Set this newly created logger as the default logger for spdlog.
    // This means that any calls to global spdlog functions like
    // spdlog::info(), spdlog::warn(), etc., will use this logger
    // and thus output to stderr.
    spdlog::set_default_logger(logger);

    // Set the logging level for the default logger.
    // For now, we'll set it to 'info' so only info, warn, error, critical
    // and off messages are shown. Debug and trace messages will be ignored.
    spdlog::set_level(spdlog::level::info); // Set default log level
}

} // namespace Utils