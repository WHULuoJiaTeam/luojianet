/**
* Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
* Copyright 2021, 2022 Huawei Technologies Co., Ltd
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
#ifndef LUOJIANET_MS_LITE_TOOLS_BENCHMARK_DPICO_SRC_CUSTOM_LOG_H_
#define LUOJIANET_MS_LITE_TOOLS_BENCHMARK_DPICO_SRC_CUSTOM_LOG_H_

#include <memory>
#include <sstream>

// NOTICE: when relative path of 'log.h' changed, macro 'DPICO_LOG_HEAR_FILE_REL_PATH' must be changed
#define DPICO_LOG_HEAR_FILE_REL_PATH "luojianet_ms/lite/tools/benchmark/dpico/src/custom_log.h"

// Get start index of file relative path in __FILE__
static constexpr size_t GetRealPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(DPICO_LOG_HEAR_FILE_REL_PATH)
           ? sizeof(__FILE__) - sizeof(DPICO_LOG_HEAR_FILE_REL_PATH)
           : 0;
}

namespace luojianet_ms {
#define DPICO_FILE_NAME                                                                         \
  (sizeof(__FILE__) > GetRealPathPos() ? static_cast<const char *>(__FILE__) + GetRealPathPos() \
                                       : static_cast<const char *>(__FILE__))

struct DpicoLocationInfo {
  DpicoLocationInfo(const char *file, int line, const char *func) : file_(file), line_(line), func_(func) {}

  ~DpicoLocationInfo() = default;

  const char *file_;
  int line_;
  const char *func_;
};

class DpicoLogStream {
 public:
  DpicoLogStream() { sstream_ = std::make_shared<std::stringstream>(); }

  ~DpicoLogStream() = default;

  template <typename T>
  DpicoLogStream &operator<<(const T &val) noexcept {
    (*sstream_) << val;
    return *this;
  }

  DpicoLogStream &operator<<(std::ostream &func(std::ostream &os)) noexcept {
    (*sstream_) << func;
    return *this;
  }
  friend class DpicoLogWriter;

 private:
  std::shared_ptr<std::stringstream> sstream_;
};

enum class DpicoLogLevel : int { DEBUG = 0, INFO, WARNING, ERROR };

class DpicoLogWriter {
 public:
  DpicoLogWriter(const DpicoLocationInfo &location, luojianet_ms::DpicoLogLevel log_level)
      : location_(location), log_level_(log_level) {}

  ~DpicoLogWriter() = default;

  __attribute__((visibility("default"))) void operator<(const DpicoLogStream &stream) const noexcept;

 private:
  void OutputLog(const std::ostringstream &msg) const;

  DpicoLocationInfo location_;
  DpicoLogLevel log_level_;
};

#define MSLOG_IF(level)                                                                                     \
  luojianet_ms::DpicoLogWriter(luojianet_ms::DpicoLocationInfo(DPICO_FILE_NAME, __LINE__, __FUNCTION__), level) < \
    luojianet_ms::DpicoLogStream()

#define MS_LOG(level) MS_LOG_##level

#define MS_LOG_DEBUG MSLOG_IF(luojianet_ms::DpicoLogLevel::DEBUG)
#define MS_LOG_INFO MSLOG_IF(luojianet_ms::DpicoLogLevel::INFO)
#define MS_LOG_WARNING MSLOG_IF(luojianet_ms::DpicoLogLevel::WARNING)
#define MS_LOG_ERROR MSLOG_IF(luojianet_ms::DpicoLogLevel::ERROR)
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_TOOLS_BENCHMARK_DPICO_SRC_CUSTOM_LOG_H_
