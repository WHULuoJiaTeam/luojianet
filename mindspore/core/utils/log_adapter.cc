/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils/log_adapter.h"

#ifndef _MSC_VER
#include <unistd.h>
#include <sys/time.h>
#endif
#include <map>
#include <iomanip>
#include <thread>
#include <vector>
#include "utils/convert_utils_base.h"

// namespace to support utils module definition
namespace mindspore {
constexpr int kNameMaxLength = 18;
std::map<void **, std::thread *> acl_handle_map;
// set default log level to WARNING for all sub modules
int g_ms_submodule_log_levels[NUM_SUBMODUES] = {WARNING};
#if defined(_WIN32) || defined(_WIN64)
enum MsLogLevel this_thread_max_log_level = EXCEPTION;
#else
thread_local enum MsLogLevel this_thread_max_log_level = EXCEPTION;
#endif

#ifdef USE_GLOG
#define google mindspore_private
static std::string GetProcName() {
#if defined(__APPLE__) || defined(__FreeBSD__)
  const std::string appname = getprogname();
#elif defined(_GNU_SOURCE)
  const std::string appname = program_invocation_name;
#else
  const std::string appname = "?";
#endif
  // sometimes, the app name is an absolute path, it is too long
  std::string app_name(appname);
  std::size_t pos = app_name.rfind("/");
  if (pos == std::string::npos) {
    return app_name;
  }
  if (pos + 1 >= app_name.size()) {
    return app_name;
  }
  return app_name.substr(pos + 1);
}

static std::string GetLogLevel(MsLogLevel level) {
  static const char *const level_names[] = {
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
  };
  if (level > this_thread_max_log_level) {
    level = this_thread_max_log_level;
  }
  return std::string(level_names[level]);
}

// convert MsLogLevel to corresponding glog level
static int GetGlogLevel(MsLogLevel level) {
  switch (level >= this_thread_max_log_level ? this_thread_max_log_level : level) {
    case DEBUG:
    case INFO:
      return google::GLOG_INFO;
    case WARNING:
      return google::GLOG_WARNING;
    case ERROR:
    case EXCEPTION:
    default:
      return google::GLOG_ERROR;
  }
}

// get threshold level
static int GetThresholdLevel(const std::string &threshold) {
  if (threshold.empty()) {
    return google::GLOG_WARNING;
  } else if (threshold == "DEBUG" || threshold == "INFO") {
    return google::GLOG_INFO;
  } else if (threshold == "WARNING") {
    return google::GLOG_WARNING;
  } else if (threshold == "ERROR" || threshold == "CRITICAL") {
    return google::GLOG_ERROR;
  } else {
    return google::GLOG_WARNING;
  }
}
#undef google
#else

#undef Dlog
#define Dlog(module_id, level, format, ...)                   \
  do {                                                        \
    DlogInner((module_id), (level), (format), ##__VA_ARGS__); \
  } while (0)

// convert MsLogLevel to corresponding slog level
static int GetSlogLevel(MsLogLevel level) {
  switch (level) {
    case DEBUG:
      return DLOG_DEBUG;
    case INFO:
      return DLOG_INFO;
    case WARNING:
      return DLOG_WARN;
    case ERROR:
    case EXCEPTION:
    default:
      return DLOG_ERROR;
  }
}
#endif

void LogWriter::set_exception_handler(const ExceptionHandler &exception_handler) {
  exception_handler_ = exception_handler;
}

void LogWriter::set_trace_provider(const TraceProvider &trace_provider) {
  if (trace_provider_ == nullptr) {
    trace_provider_ = trace_provider;
  } else {
    MS_LOG(INFO) << "trace provider has been set, skip.";
  }
}

LogWriter::TraceProvider LogWriter::trace_provider() { return trace_provider_; }

void LogWriter::OutputLog(const std::ostringstream &msg) const {
#ifdef USE_GLOG
#define google mindspore_private
  auto submodule_name = GetSubModuleName(submodule_);
  google::LogMessage("", 0, GetGlogLevel(log_level_)).stream()
#ifdef _MSC_VER
    << "[" << GetLogLevel(log_level_) << "] " << submodule_name << "("
    << "," << std::hex
#else
    << "[" << GetLogLevel(log_level_) << "] " << submodule_name << "(" << getpid() << "," << std::hex
#endif
    << std::this_thread::get_id() << std::dec << "," << GetProcName() << "):" << GetTimeString() << " "
    << "[" << location_.file_ << ":" << location_.line_ << "] " << location_.func_ << "] " << msg.str() << std::endl;
#undef google
#else
  auto str_msg = msg.str();
  auto slog_module_id = (submodule_ == SM_MD ? MD : ME);
  Dlog(static_cast<int>(slog_module_id), GetSlogLevel(log_level_), "[%s:%d] %s] %s", location_.file_, location_.line_,
       location_.func_, str_msg.c_str());
#endif
}

void LogWriter::operator<(const LogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}
#ifndef BUILD_LITE_INFERENCE
void LogWriter::operator^(const LogStream &stream) const {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  std::ostringstream oss;
  oss << location_.file_ << ":" << location_.line_ << " " << location_.func_ << "] ";
  oss << msg.str();

  thread_local bool running = false;
  if (!running) {
    running = true;
    if (this_thread_max_log_level >= EXCEPTION) {
      OutputLog(msg);
    }
    if (trace_provider_ != nullptr) {
      trace_provider_(oss);
    }
    running = false;
  }

  if (exception_handler_ != nullptr) {
    exception_handler_(exception_type_, oss.str());
  }
  throw std::runtime_error(oss.str());
}
#endif

static inline std::string GetEnv(const std::string &envvar) {
  const char *value = std::getenv(envvar.c_str());

  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

enum class LogConfigToken : size_t {
  INVALID,      // indicate invalid token
  LEFT_BRACE,   // '{'
  RIGHT_BRACE,  // '}'
  VARIABLE,     // '[A-Za-z][A-Za-z0-9_]*'
  NUMBER,       // [0-9]+
  COMMA,        // ','
  COLON,        // ':'
  EOS,          // End Of String, '\0'
  NUM_LOG_CFG_TOKENS
};

static const char *g_tok_names[static_cast<size_t>(LogConfigToken::NUM_LOG_CFG_TOKENS)] = {
  "invalid",        // indicate invalid token
  "{",              // '{'
  "}",              // '}'
  "variable",       // '[A-Za-z][A-Za-z0-9_]*'
  "number",         // [0-9]+
  ",",              // ','
  ":",              // ':'
  "end-of-string",  // End Of String, '\0'
};

static inline bool IsAlpha(char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); }

static inline bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

class LogConfigLexer {
 public:
  explicit LogConfigLexer(const std::string &text) : buffer_(text), cur_idx_(0) {}
  ~LogConfigLexer() = default;

  // skip white space, and return the first char after white space
  char SkipWhiteSpace() {
    while (cur_idx_ < buffer_.size()) {
      char ch = buffer_[cur_idx_];
      if (ch == ' ' || ch == '\t') {
        ++cur_idx_;
        continue;
      }
      return ch;
    }
    return '\0';
  }

  LogConfigToken GetNext(std::string *const ptr) {
#ifdef DEBUG
    std::string text;
    auto tok = GetNextInner(&text);
    MS_LOG(DEBUG) << "Got token " << tok << " with value [" << text << "]";
    if (ptr != nullptr) {
      *ptr = text;
    }
    return tok;
  }

  LogConfigToken GetNextInner(std::string *ptr) {
#endif
    char ch = SkipWhiteSpace();
    // clang-format off
    static const std::map<char, LogConfigToken> single_char_map = {
      {'{', LogConfigToken::LEFT_BRACE},
      {'}', LogConfigToken::RIGHT_BRACE},
      {',', LogConfigToken::COMMA},
      {':', LogConfigToken::COLON},
      {'\0', LogConfigToken::EOS},
    };
    // clang-format on

    auto iter = single_char_map.find(ch);
    if (iter != single_char_map.end()) {
      if (ptr != nullptr) {
        *ptr = std::string() + ch;
      }
      ++cur_idx_;
      return iter->second;
    } else if (IsAlpha(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && (IsAlpha(ch) || IsDigit(ch) || ch == '_'));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::VARIABLE;
    } else if (IsDigit(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && IsDigit(ch));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::NUMBER;
    }
    return LogConfigToken::INVALID;
  }

 private:
  std::string buffer_;
  size_t cur_idx_;
};

class LogConfigParser {
 public:
  explicit LogConfigParser(const std::string &cfg) : lexer(cfg) {}
  ~LogConfigParser() = default;

  bool Expect(LogConfigToken expected, LogConfigToken tok) const {
    if (expected != tok) {
      MS_LOG(WARNING) << "Parse submodule log configuration text error, expect `"
                      << g_tok_names[static_cast<size_t>(expected)] << "`, but got `"
                      << g_tok_names[static_cast<size_t>(tok)] << "`. The whole configuration will be ignored.";
      return false;
    }
    return true;
  }

  // The text of config MS_SUBMODULE_LOG_v is in the form {submodule1:log_level1,submodule2:log_level2,...}.
  // Valid values of log levels are: 0 - debug, 1 - info, 2 - warning, 3 - error, 4 - critical
  // e.g. MS_SUBMODULE_LOG_v={PARSER:0, ANALYZER:2, PIPELINE:1}
  std::map<std::string, std::string> Parse() {
    std::map<std::string, std::string> log_levels;

    bool flag_error = false;
    std::string text;
    auto tok = lexer.GetNext(&text);
    // empty string
    if (tok == LogConfigToken::EOS) {
      return log_levels;
    }

    if (!Expect(LogConfigToken::LEFT_BRACE, tok)) {
      return log_levels;
    }

    do {
      std::string key, val;
      tok = lexer.GetNext(&key);
      if (!Expect(LogConfigToken::VARIABLE, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&text);
      if (!Expect(LogConfigToken::COLON, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&val);
      if (!Expect(LogConfigToken::NUMBER, tok)) {
        flag_error = true;
        break;
      }

      log_levels[key] = val;
      tok = lexer.GetNext(&text);
    } while (tok == LogConfigToken::COMMA);

    if (!flag_error && !Expect(LogConfigToken::RIGHT_BRACE, tok)) {
      flag_error = true;
    }

    if (flag_error) {
      log_levels.clear();
    }
    return log_levels;
  }

 private:
  LogConfigLexer lexer;
};

bool ParseLogLevel(const std::string &str_level, MsLogLevel *ptr_level) {
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    constexpr char number_start = '0';
    ch = ch - number_start;  // subtract ASCII code of '0', which is 48
    if (ch >= static_cast<int>(DEBUG) && ch <= static_cast<int>(EXCEPTION)) {
      if (ptr_level != nullptr) {
        *ptr_level = static_cast<MsLogLevel>(ch);
      }
      return true;
    }
  }
  return false;
}

static MsLogLevel GetGlobalLogLevel() {
#ifdef USE_GLOG
  return static_cast<MsLogLevel>(FLAGS_v);
#else
  int log_level = WARNING;  // set default log level to WARNING
  auto str_level = GetEnv("GLOG_v");
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    constexpr char number_start = '0';
    ch = ch - number_start;  // subtract ASCII code of '0', which is 48
    if (ch >= DEBUG && ch <= EXCEPTION) {
      log_level = ch;
    }
  }
  return static_cast<MsLogLevel>(log_level);
#endif
}

void InitSubModulesLogLevel() {
  // initialize submodule's log level using global
  auto global_log_level = GetGlobalLogLevel();
  for (int i = 0; i < static_cast<int>(NUM_SUBMODUES); ++i) {
    g_ms_submodule_log_levels[i] = global_log_level;
  }

  // set submodule's log level
  auto submodule = GetEnv("MS_SUBMODULE_LOG_v");
  MS_LOG(DEBUG) << "MS_SUBMODULE_LOG_v=`" << submodule << "`";
  LogConfigParser parser(submodule);
  auto configs = parser.Parse();
  for (const auto &cfg : configs) {
    int mod_idx = -1;
    for (int i = 0; i < static_cast<int>(NUM_SUBMODUES); ++i) {
      if (cfg.first == GetSubModuleName(static_cast<SubModuleId>(i))) {
        mod_idx = i;
        break;
      }
    }
    if (mod_idx < 0) {
      MS_LOG(WARNING) << "Undefined module name " << cfg.first << ", ignore it";
      continue;
    }
    MsLogLevel submodule_log_level;
    if (!ParseLogLevel(cfg.second, &submodule_log_level)) {
      MS_LOG(WARNING) << "Illegal log level value " << cfg.second << " for " << cfg.first << ", ignore it.";
      continue;
    }
    g_ms_submodule_log_levels[mod_idx] = static_cast<int>(submodule_log_level);
  }
}

const std::string GetSubModuleName(SubModuleId module_id) {
  static const char sub_module_names[NUM_SUBMODUES][kNameMaxLength] = {
    "UNKNOWN",            // SM_UNKNOWN
    "CORE",               // SM_CORE
    "ANALYZER",           // SM_ANALYZER
    "COMMON",             // SM_COMMON
    "DEBUG",              // SM_DEBUG
    "OFFLINE_DEBUG",      // SM_OFFLINE_DEBUG
    "DEVICE",             // SM_DEVICE
    "GE_ADPT",            // SM_GE_ADPT
    "IR",                 // SM_IR
    "KERNEL",             // SM_KERNEL
    "MD",                 // SM_MD
    "ME",                 // SM_ME
    "EXPRESS",            // SM_EXPRESS
    "OPTIMIZER",          // SM_OPTIMIZER
    "PARALLEL",           // SM_PARALLEL
    "PARSER",             // SM_PARSER
    "PIPELINE",           // SM_PIPELINE
    "PRE_ACT",            // SM_PRE_ACT
    "PYNATIVE",           // SM_PYNATIVE
    "SESSION",            // SM_SESSION
    "UTILS",              // SM_UTILS
    "VM",                 // SM_VM
    "PROFILER",           // SM_PROFILER
    "PS",                 // SM_PS
    "FL",                 // SM_FL
    "DISTRIBUTED",        // SM_DISTRIBUTED
    "LITE",               // SM_LITE
    "ARMOUR",             // SM_ARMOUR
    "HCCL_ADPT",          // SM_HCCL_ADPT
    "RUNTIME_FRAMEWORK",  // SM_RUNTIME_FRAMEWORK
    "GE",                 // SM_GE
    "API",                // SM_API
  };
  return sub_module_names[IntToSize(module_id % NUM_SUBMODUES)];
}

std::string GetTimeString() {
  constexpr auto BUFLEN = 80;
  char buf[BUFLEN] = {'\0'};
#if defined(_WIN32) || defined(_WIN64)
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);
  constexpr int base_year = 1900;
  (void)snprintf(buf, BUFLEN, "%d-%d-%d %d:%d:%d", now_time.tm_year + base_year, now_time.tm_mon + 1, now_time.tm_mday,
                 now_time.tm_hour, now_time.tm_min, now_time.tm_sec);
#else
  struct timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);

  struct tm now;
  constexpr size_t time_str_len = 19;
  constexpr int64_t time_convert_unit = 1000;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, BUFLEN, "%Y-%m-%d-%H:%M:%S", &now);  // format date and time
#ifdef __APPLE__
  constexpr auto fmt_str = ".%03lld.%03lld";
#else
  constexpr auto fmt_str = ".%03ld.%03ld";
#endif
  (void)snprintf(buf + time_str_len, BUFLEN - time_str_len, fmt_str, cur_time.tv_usec / time_convert_unit,
                 cur_time.tv_usec % time_convert_unit);
#endif
  return std::string(buf);
}
}  // namespace mindspore

extern "C" {
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#ifdef _MSC_VER
MS_CORE_API void common_log_init(void) {
#else
__attribute__((constructor)) MS_CORE_API void common_log_init(void) {
#endif
#else
MS_CORE_API void common_log_init(void) {
#endif
#ifdef USE_GLOG
  // Do not use glog predefined log prefix
  FLAGS_log_prefix = false;
  // Write log to files real-time
  FLAGS_logbufsecs = 0;
  // Set default log level to WARNING
  if (mindspore::GetEnv("GLOG_v").empty()) {
    FLAGS_v = static_cast<int>(mindspore::WARNING);
  }

  // Set default log file mode to 0640
  if (mindspore::GetEnv("GLOG_logfile_mode").empty()) {
    FLAGS_logfile_mode = 0640;
  }
  // Set default log file max size to 50 MB
  FLAGS_max_log_size = 50;
  std::string max_log_size = mindspore::GetEnv("GLOG_max_log_size");
  if (!max_log_size.empty()) {
    FLAGS_max_log_size = std::stoi(max_log_size);
  }
  std::string logtostderr = mindspore::GetEnv("GLOG_logtostderr");
  // Default print log to screen
  if (logtostderr.empty()) {
    FLAGS_logtostderr = true;
  } else if (logtostderr == "0") {
    if (mindspore::GetEnv("GLOG_log_dir").empty()) {
#ifndef BUILD_LITE_INFERENCE
      MS_LOG(ERROR) << "`GLOG_log_dir` is empty, it must be set while 'logtostderr' equals to 0.";
      // Here can not throw exception and use python to catch, because the PYBIND11_MODULE is not yet been initialed.
      exit(EXIT_FAILURE);
#else
      MS_LOG(WARNING) << "`GLOG_log_dir` is empty, log will be printed to stderr.";
      FLAGS_logtostderr = true;
#endif
    } else {
      // Set log dir from GLOG_log_dir with RANK_ID or OMPI_COMM_WORLD_RANK.
#ifndef BUILD_LITE_INFERENCE
      const std::string rank_id = mindspore::GetEnv("RANK_ID");
      const std::string gpu_rank_id = mindspore::GetEnv("OMPI_COMM_WORLD_RANK");
      std::string rank = "0";
      if ((!rank_id.empty() && gpu_rank_id.empty()) || (!rank_id.empty() && !gpu_rank_id.empty())) {
        rank = rank_id;
      } else if (rank_id.empty() && !gpu_rank_id.empty()) {
        rank = gpu_rank_id;
      }
      FLAGS_log_dir = mindspore::GetEnv("GLOG_log_dir") + "/rank_" + rank + "/logs";
#else
      FLAGS_log_dir = mindspore::GetEnv("GLOG_log_dir");
#endif
    }
  }

  // Default GLOG_stderrthreshold level to WARNING
  auto threshold = mindspore::GetEnv("GLOG_stderrthreshold");
  FLAGS_stderrthreshold = mindspore::GetThresholdLevel(threshold);

#endif
  mindspore::InitSubModulesLogLevel();
}

// shared lib init hook
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#ifdef _MSC_VER
void mindspore_log_init(void) {
#else
__attribute__((constructor)) void mindspore_log_init(void) {
#endif
#else
void mindspore_log_init(void) {
#endif
#ifdef USE_GLOG
#define google mindspore_private
  static bool is_glog_initialzed = false;
  if (!is_glog_initialzed) {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
    google::InitGoogleLogging("mindspore");
#endif
    is_glog_initialzed = true;
  }
#undef google
#endif
  common_log_init();
}
}
