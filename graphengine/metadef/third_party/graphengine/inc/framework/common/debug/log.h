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

#ifndef INC_FRAMEWORK_COMMON_DEBUG_LOG_H_
#define INC_FRAMEWORK_COMMON_DEBUG_LOG_H_

#include <string>
#include <sstream>

#include "framework/common/string_util.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "external/ge/ge_api_error_codes.h"

#if !defined(__ANDROID__) && !defined(ANDROID)
#define DOMI_LOGE(fmt, ...) GE_LOG_ERROR(GE_MODULE_NAME, ge::FAILED, fmt, ##__VA_ARGS__)
#else
#include <android/log.h>
#if defined(BUILD_VERSION_PERF)
#define DOMI_LOGE(fmt, ...)
#else
// The Android system has strict log control. Do not modify the log.
#define DOMI_LOGE(fmt, ...) \
  __android_log_print(ANDROID_LOG_ERROR, "NPU_FMK", "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif
#endif

// ge marco
#define GE_LOGI_IF(condition, ...) \
  if ((condition)) {               \
    GELOGI(__VA_ARGS__);           \
  }

#define GE_LOGW_IF(condition, ...) \
  if ((condition)) {               \
    GELOGW(__VA_ARGS__);           \
  }

#define GE_LOGE_IF(condition, ...)   \
  if ((condition)) {                 \
    GELOGE(ge::FAILED, __VA_ARGS__); \
  }

// If expr is not SUCCESS, print the log and return the same value
#define GE_CHK_STATUS_RET(expr, ...)        \
  do {                                      \
    const ge::Status _chk_status = (expr);  \
    if (_chk_status != ge::SUCCESS) {       \
      GELOGE(ge::FAILED, __VA_ARGS__);      \
      return _chk_status;                   \
    }                                       \
  } while (false)

// If expr is not SUCCESS, print the log and do not execute return
#define GE_CHK_STATUS(expr, ...)            \
  do {                                      \
    const ge::Status _chk_status = (expr);  \
    if (_chk_status != ge::SUCCESS) {       \
      GELOGE(ge::FAILED, __VA_ARGS__);      \
    }                                       \
  } while (false)

// If expr is not SUCCESS, return the same value
#define GE_CHK_STATUS_RET_NOLOG(expr)       \
  do {                                      \
    const ge::Status _chk_status = (expr);  \
    if (_chk_status != ge::SUCCESS) {       \
      return _chk_status;                   \
    }                                       \
  } while (false)

// If expr is not GRAPH_SUCCESS, print the log and return FAILED
#define GE_CHK_GRAPH_STATUS_RET(expr, ...)                  \
  do {                                                      \
    if ((expr) != ge::GRAPH_SUCCESS) {                      \
      REPORT_CALL_ERROR("E19999", "Operator graph failed"); \
      GELOGE(ge::FAILED, __VA_ARGS__);                      \
      return FAILED;                                        \
    }                                                       \
  } while (false)

// If expr is not SUCCESS, print the log and execute a custom statement
#define GE_CHK_STATUS_EXEC(expr, exec_expr, ...)                      \
  do {                                                                \
    const ge::Status _chk_status = (expr);                            \
    GE_CHK_BOOL_EXEC(_chk_status == SUCCESS, exec_expr, __VA_ARGS__); \
  } while (false)

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...) \
  do {                                             \
    const bool b = (expr);                         \
    if (!b) {                                      \
      REPORT_INNER_ERROR("E19999", __VA_ARGS__);   \
      GELOGE(_status, __VA_ARGS__);                \
      return _status;                              \
    }                                              \
  } while (false)

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS_NOLOG(expr, _status, ...) \
  do {                                                   \
    const bool b = (expr);                               \
    if (!b) {                                            \
      return _status;                                    \
    }                                                    \
  } while (false)

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC(expr, exec_expr, ...) \
  {                                            \
    const bool b = (expr);                     \
    if (!b) {                                  \
      GELOGE(ge::FAILED, __VA_ARGS__);         \
      exec_expr;                               \
    }                                          \
  }

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC_WARN(expr, exec_expr, ...) \
  {                                                 \
    const bool b = (expr);                          \
    if (!b) {                                       \
      GELOGW(__VA_ARGS__);                          \
      exec_expr;                                    \
    }                                               \
  }
// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC_INFO(expr, exec_expr, ...) \
  {                                                 \
    const bool b = (expr);                          \
    if (!b) {                                       \
      GELOGI(__VA_ARGS__);                          \
      exec_expr;                                    \
    }                                               \
  }

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_TRUE_EXEC_INFO(expr, exec_expr, ...) \
  {                                                      \
    const bool b = (expr);                               \
    if (b) {                                             \
      GELOGI(__VA_ARGS__);                               \
      exec_expr;                                         \
    }                                                    \
  }

// If expr is true, print logs and execute custom statements
#define GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(expr, exec_expr, ...) \
  {                                                          \
    const bool b = (expr);                                   \
    if (b) {                                                 \
      GELOGE(ge::FAILED, __VA_ARGS__);                       \
      exec_expr;                                             \
    }                                                        \
  }
// If expr is true, print the Information log and execute a custom statement
#define GE_CHK_TRUE_EXEC_INFO(expr, exec_expr, ...) \
  {                                                 \
    const bool b = (expr);                          \
    if (b) {                                        \
      GELOGI(__VA_ARGS__);                          \
      exec_expr;                                    \
    }                                               \
  }

// If expr is not SUCCESS, print the log and execute the expression + return
#define GE_CHK_BOOL_TRUE_RET_VOID(expr, exec_expr, ...) \
  {                                                     \
    const bool b = (expr);                              \
    if (b) {                                            \
      GELOGE(ge::FAILED, __VA_ARGS__);                  \
      exec_expr;                                        \
      return;                                           \
    }                                                   \
  }

// If expr is not SUCCESS, print the log and execute the expression + return _status
#define GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(expr, _status, exec_expr, ...) \
  {                                                                     \
    const bool b = (expr);                                              \
    if (b) {                                                            \
      REPORT_INNER_ERROR("E19999", __VA_ARGS__);                        \
      GELOGE(ge::FAILED, __VA_ARGS__);                                  \
      exec_expr;                                                        \
      return _status;                                                   \
    }                                                                   \
  }

// If expr is not true, execute a custom statement
#define GE_CHK_BOOL_EXEC_NOLOG(expr, exec_expr) \
  {                                             \
    const bool b = (expr);                      \
    if (!b) {                                   \
      exec_expr;                                \
    }                                           \
  }

// -----------------runtime related macro definitions-------------------------------
// If expr is not RT_ERROR_NONE, print the log
#define GE_CHK_RT(expr)                                             \
  do {                                                              \
    const rtError_t _rt_ret = (expr);                               \
    if (_rt_ret != RT_ERROR_NONE) {                                 \
      GELOGE(ge::FAILED, "Call rt api failed, ret: 0x%X", _rt_ret); \
    }                                                               \
  } while (false)

// If expr is not RT_ERROR_NONE, print the log and execute the exec_expr expression
#define GE_CHK_RT_EXEC(expr, exec_expr)                             \
  do {                                                              \
    const rtError_t _rt_ret = (expr);                               \
    if (_rt_ret != RT_ERROR_NONE) {                                 \
      GELOGE(ge::FAILED, "Call rt api failed, ret: 0x%X", _rt_ret); \
      exec_expr;                                                    \
    }                                                               \
  } while (false)

// If expr is not RT_ERROR_NONE, print the log and return
#define GE_CHK_RT_RET(expr)                                                   \
  do {                                                                        \
    const rtError_t _rt_ret = (expr);                                         \
    if (_rt_ret != RT_ERROR_NONE) {                                           \
      REPORT_CALL_ERROR("E19999", "Call %s fail, ret: 0x%X", #expr, _rt_ret); \
      GELOGE(ge::FAILED, "Call rt api failed, ret: 0x%X", _rt_ret);           \
      return RT_ERROR_TO_GE_STATUS(_rt_ret);                                  \
    }                                                                         \
  } while (false)

// If expr is true, execute exec_expr without printing logs
#define GE_IF_BOOL_EXEC(expr, exec_expr) \
  {                                      \
    if (expr) {                          \
      exec_expr;                         \
    }                                    \
  }

// If make_shared is abnormal, print the log and execute the statement
#define GE_MAKE_SHARED(exec_expr0, exec_expr1) \
  try {                                        \
    exec_expr0;                                \
  } catch (const std::bad_alloc &) {           \
    GELOGE(ge::FAILED, "Make shared failed");  \
    exec_expr1;                                \
  }

#define GE_ERRORLOG_AND_ERRORMSG(_status, errormsg)    \
  {                                                    \
    GELOGE(_status, "[Check][InnerData]%s", errormsg); \
    REPORT_INNER_ERROR("E19999", "%s", errormsg);      \
  }

#define GE_WARNINGLOG_AND_ERRORMSG(errormsg)                                           \
  {                                                                                    \
    GELOGW("%s", errormsg);                                                            \
    ErrorManager::GetInstance().ATCReportErrMessage("E19021", {"reason"}, {errormsg}); \
  }

#define GE_CHK_LOG_AND_ERRORMSG(expr, _status, errormsg)                                 \
  do {                                                                                   \
    const bool b = (expr);                                                               \
    if (!b) {                                                                            \
      GELOGE(_status, "%s", errormsg);                                                   \
      ErrorManager::GetInstance().ATCReportErrMessage("E19021", {"reason"}, {errormsg}); \
      return _status;                                                                    \
    }                                                                                    \
  } while (false)

template <typename T>
GE_FUNC_VISIBILITY std::string FmtToStr(const T &t) {
  std::string fmt;
  std::stringstream st;
  st << "[" << t << "]";
  fmt = st.str();
  return fmt;
}

#endif  // INC_FRAMEWORK_COMMON_DEBUG_LOG_H_
