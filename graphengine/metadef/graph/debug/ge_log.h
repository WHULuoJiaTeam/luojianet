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

#ifndef COMMON_GRAPH_DEBUG_GE_LOG_H_
#define COMMON_GRAPH_DEBUG_GE_LOG_H_

#include "graph/ge_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"

#define GE_LOGE(fmt, ...) GE_LOG_ERROR(GE_MODULE_NAME, ge::FAILED, fmt, ##__VA_ARGS__)

// Only check error log
#define GE_CHK_BOOL_ONLY_LOG(expr, ...) \
  do {                                  \
    const bool b = (expr);              \
    if (!b) {                           \
      GELOGI(__VA_ARGS__);              \
    }                                   \
  } while (false)

#endif  // COMMON_GRAPH_DEBUG_GE_LOG_H_

