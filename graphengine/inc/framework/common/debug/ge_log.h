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

#ifndef INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_
#define INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_

#include <cstdint>

#include "framework/common/ge_inner_error_codes.h"
#include "common/util/error_manager/error_manager.h"
#include "toolchain/slog.h"
#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#else
#include "mmpa/mmpa_api.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GE_MODULE_NAME static_cast<int32_t>(GE)

// trace status of log
enum TraceStatus { TRACE_INIT = 0, TRACE_RUNNING, TRACE_WAITING, TRACE_STOP };

class GE_FUNC_VISIBILITY GeLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  const int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  return (enable == 1);
}

#define GELOGE(ERROR_CODE, fmt, ...)                                                                              \
  do {                                                                                                            \
    dlog_error(GE_MODULE_NAME, "%lu %s: ErrorNo: %u(%s) %s" fmt, GeLog::GetTid(), &__FUNCTION__[0], (ERROR_CODE), \
               ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ErrorManager::GetInstance().GetLogHeader().c_str(),    \
               ##__VA_ARGS__);                                                                                    \
  } while (false)

#define GELOGW(fmt, ...)                                                                          \
  do {                                                                                            \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_WARN)) {                                                 \
      dlog_warn(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0], ##__VA_ARGS__); \
    }                                                                                             \
  } while (false)

#define GELOGI(fmt, ...)                                                                          \
  do {                                                                                            \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {                                                 \
      dlog_info(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0], ##__VA_ARGS__); \
    }                                                                                             \
  } while (false)

#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {                                                 \
      dlog_debug(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0], ##__VA_ARGS__); \
    }                                                                                              \
  } while (false)

#define GEEVENT(fmt, ...)                                                                        \
  do {                                                                                           \
    dlog_event(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0], ##__VA_ARGS__); \
  } while (false)

#define GELOGT(VALUE, fmt, ...)                                                                                      \
  do {                                                                                                               \
    TraceStatus stat = (VALUE);                                                                                      \
    const char_t *const TraceStatStr[] = {"INIT", "RUNNING", "WAITING", "STOP"};                                     \
    const int32_t idx = static_cast<int32_t>(stat);                                                                  \
    char_t *k = const_cast<char_t *>("status");                                                                      \
    char_t *v = const_cast<char_t *>(TraceStatStr[idx]);                                                             \
    KeyValue kv = {k, v};                                                                                            \
    DlogWithKV(GE_MODULE_NAME, DLOG_TRACE, &kv, 1, "%lu %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0], ##__VA_ARGS__); \
  } while (false)

#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)                                                           \
  do {                                                                                                         \
    dlog_error((MOD_NAME), "%lu %s: ErrorNo: %u(%s) %s" fmt, GeLog::GetTid(), &__FUNCTION__[0], (ERROR_CODE),  \
               ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ErrorManager::GetInstance().GetLogHeader().c_str(), \
               ##__VA_ARGS__);                                                                                 \
  } while (false)

// print memory when it is greater than 1KB.
#define GE_PRINT_DYNAMIC_MEMORY(FUNC, PURPOSE, SIZE)                                                        \
  do {                                                                                                      \
    if (static_cast<size_t>(SIZE) > 1024UL) {                                                               \
      GELOGI("MallocMemory, func=%s, size=%zu, purpose=%s", (#FUNC), static_cast<size_t>(SIZE), (PURPOSE)); \
    }                                                                                                       \
  } while (false)
#ifdef __cplusplus
}
#endif
#endif  // INC_FRAMEWORK_COMMON_DEBUG_GE_LOG_H_
