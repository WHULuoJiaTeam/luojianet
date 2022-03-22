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

#ifndef INC_GRAPH_GRAPH_UTIL_H_
#define INC_GRAPH_GRAPH_UTIL_H_

#include <string>

#include "proto/om.pb.h"

namespace ge {
using AttrDefMap = ::google::protobuf::Map<::std::string, ::domi::AttrDef>;
bool HasOpAttr(const OpDef *opdef, std::string attr_name);
bool GetOpAttr(const std::string &key, int32_t *value, const OpDef *opdef);

static const char OP_TYPE_DATA[] = "Data";
static const char OP_TYPE_INPUT[] = "Input";
static const char ATTR_KEY_INPUT_FORMAT[] = "input_format";
static const char ATTR_KEY_OUTPUT_FORMAT[] = "output_format";
static const char OP_TYPE_ANN_DATA[] = "AnnData";
}  // namespace ge

#if !defined(__ANDROID__) && !defined(ANDROID)
#include "toolchain/slog.h"
const char levelStr[4][8] = {"ERROR", "WARN", "INFO", "DEBUG"};
#else
#include <syslog.h>
#include <utils/Log.h>
const char levelStr[8][8] = {"EMERG", "ALERT", "CRIT", "ERROR", "WARNING", "NOTICE", "INFO", "DEBUG"};
#endif

#ifdef _MSC_VER
#define FUNC_NAME __FUNCTION__
#else
#define FUNC_NAME __PRETTY_FUNCTION__
#endif

#if !defined(__ANDROID__) && !defined(ANDROID)
#define D_GRAPH_LOGI(MOD_NAME, fmt, ...) \
  dlog_info(FMK, "%s:%s:%d:" #fmt, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define D_GRAPH_LOGW(MOD_NAME, fmt, ...) \
  dlog_warn(FMK, "%s:%s:%d:" #fmt, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define D_GRAPH_LOGE(MOD_NAME, fmt, ...) \
  dlog_error(FMK, "%s:%s:%d:" #fmt, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define D_GRAPH_LOG(level, format, ...)                                                                       \
  do {                                                                                                        \
    {                                                                                                         \
      fprintf(stdout, "[%s] [%s] [%s] [%s] [%s:%d] " format "\n", "", "GRAPH", levelStr[level], __FUNCTION__, \
              __FILE__, __LINE__, ##__VA_ARGS__);                                                             \
      syslog(level, "%s %s:%d] [%s] %s " format "\n", "", __FILE__, __LINE__, "OPTIMIZER", __FUNCTION__,      \
             ##__VA_ARGS__);                                                                                  \
    }                                                                                                         \
  } while (0)
#define D_GRAPH_LOGI(MOD_NAME, fmt, ...) D_GRAPH_LOG(ANDROID_LOG_INFO, #fmt, ##__VA_ARGS__)
#define D_GRAPH_LOGW(MOD_NAME, fmt, ...) D_GRAPH_LOG(ANDROID_LOG_INFO, #fmt, ##__VA_ARGS__)
#define D_GRAPH_LOGE(MOD_NAME, fmt, ...) D_GRAPH_LOG(ANDROID_LOG_INFO, #fmt, ##__VA_ARGS__)
#endif

#if !defined(__ANDROID__) && !defined(ANDROID)
#define GRAPH_LOGI(...) D_GRAPH_LOGI(GRAPH_MOD_NAME, __VA_ARGS__)
#define GRAPH_LOGW(...) D_GRAPH_LOGW(GRAPH_MOD_NAME, __VA_ARGS__)
#define GRAPH_LOGE(...) D_GRAPH_LOGE(GRAPH_MOD_NAME, __VA_ARGS__)
#else

#define GRAPH_LOG(level, format, ...)                                                                         \
  do {                                                                                                        \
    {                                                                                                         \
      fprintf(stdout, "[%s] [%s] [%s] [%s] [%s:%d] " format "\n", "", "GRAPH", levelStr[level], __FUNCTION__, \
              __FILE__, __LINE__, ##__VA_ARGS__);                                                             \
      syslog(level, "%s %s:%d] [%s] %s " format "\n", "", __FILE__, __LINE__, "OPTIMIZER", __FUNCTION__,      \
             ##__VA_ARGS__);                                                                                  \
    }                                                                                                         \
  } while (0)
#define GRAPH_LOGI(fmt, ...) GRAPH_LOG(ANDROID_LOG_INFO, #fmt, ##__VA_ARGS__)
#define GRAPH_LOGW(fmt, ...) GRAPH_LOG(ANDROID_LOG_INFO, #fmt, ##__VA_ARGS__)
#define GRAPH_LOGE(fmt, ...) GRAPH_LOG(ANDROID_LOG_INFO, #fmt, ##__VA_ARGS__)
#endif

#define GRAPH_CHK_STATUS_RET_NOLOG(expr)      \
  do {                                        \
    const domi::graphStatus _status = (expr); \
    if (_status != domi::GRAPH_SUCCESS) {     \
      return _status;                         \
    }                                         \
  } while (0)

#define GRAPH_CHK_BOOL_RET_STATUS(expr, _status, ...) \
  do {                                                \
    bool b = (expr);                                  \
    if (!b) {                                         \
      GRAPH_LOGE(__VA_ARGS__);                        \
      return _status;                                 \
    }                                                 \
  } while (0)

#define GRAPH_CHK_BOOL_EXEC_NOLOG(expr, exec_expr) \
  {                                                \
    bool b = (expr);                               \
    if (!b) {                                      \
      exec_expr;                                   \
    }                                              \
  };

#define GRAPH_IF_BOOL_EXEC(expr, exec_expr) \
  {                                         \
    if (expr) {                             \
      exec_expr;                            \
    }                                       \
  }

#define GRAPH_RETURN_WITH_LOG_IF_ERROR(expr, ...) \
  do {                                            \
    const ::domi::graphStatus _status = (expr);   \
    if (_status) {                                \
      GRAPH_LOGE(__VA_ARGS__);                    \
      return _status;                             \
    }                                             \
  } while (0)

#endif  // INC_GRAPH_GRAPH_UTIL_H_
