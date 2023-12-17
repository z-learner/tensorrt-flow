#pragma once

#include "NvInfer.h"

#include <memory>
#include <stdarg.h>
#include <string>

#define LOG_FATAL(...) tensorrt_flow::logger::Logger::__log_info(tensorrt_flow::logger::Level::FATAL, __VA_ARGS__)
#define LOG_ERROR(...) tensorrt_flow::logger::Logger::__log_info(tensorrt_flow::logger::Level::ERROR, __VA_ARGS__)
#define LOG_WARN(...) tensorrt_flow::logger::Logger::__log_info(tensorrt_flow::logger::Level::WARN, __VA_ARGS__)
#define LOG_INFO(...) tensorrt_flow::logger::Logger::__log_info(tensorrt_flow::logger::Level::INFO, __VA_ARGS__)
#define LOG_VERB(...) tensorrt_flow::logger::Logger::__log_info(tensorrt_flow::logger::Level::VERB, __VA_ARGS__)
#define LOG_DEBUG(...) tensorrt_flow::logger::Logger::__log_info(tensorrt_flow::logger::Level::DEBUG, __VA_ARGS__)

#define DGREEN "\033[1;36m"
#define BLUE "\033[1;34m"
#define PURPLE "\033[1;35m"
#define GREEN "\033[1;32m"
#define YELLOW "\033[1;33m"
#define RED "\033[1;31m"
#define CLEAR "\033[0m"

namespace tensorrt_flow {

namespace logger {

enum class Level : int32_t { FATAL = 0, ERROR = 1, WARN = 2, INFO = 3, VERB = 4, DEBUG = 5 };

class Logger : public nvinfer1::ILogger {

public:
  Logger();
  Logger(Level level);
  virtual void log(Severity severity, const char* msg) noexcept override;
  static void  __log_info(Level level, const char* format, ...);
  Severity     get_severity(Level level);
  Level        get_level(Severity severity);

private:
  static Level m_level;
  Severity     m_severity;
};

std::shared_ptr<Logger> create_logger(Level level);

}   // namespace logger
}   // namespace tensorrt_flow
