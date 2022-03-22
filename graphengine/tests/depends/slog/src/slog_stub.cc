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

#include "toolchain/slog.h"
#include "toolchain/plog.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void dav_log(int module_id, const char *fmt, ...) {}

static int log_level = DLOG_ERROR;

#define __DO_PRINT()                                 \
  do {                                               \
    const int FMT_BUFF_SIZE = 1024;                  \
    char fmt_buff[FMT_BUFF_SIZE] = {0};              \
    va_list valist;                                  \
    va_start(valist, fmt);                           \
    vsnprintf(fmt_buff, FMT_BUFF_SIZE, fmt, valist); \
    va_end(valist);                                  \
    printf("%s \n", fmt_buff);                       \
  } while (0)

void DlogErrorInner(int module_id, const char *fmt, ...) {
  if (log_level > DLOG_ERROR) {
    return;
  }
  __DO_PRINT();
}

void DlogWarnInner(int module_id, const char *fmt, ...) {
  if (log_level > DLOG_WARN) {
    return;
  }
  __DO_PRINT();
}

void DlogInfoInner(int module_id, const char *fmt, ...) {
  if (log_level > DLOG_INFO) {
    return;
  }
  __DO_PRINT();
}

void DlogDebugInner(int module_id, const char *fmt, ...) {
  if (log_level > DLOG_DEBUG) {
    return;
  }
  __DO_PRINT();
}

void DlogEventInner(int module_id, const char *fmt, ...) { dav_log(module_id, fmt); }

void DlogInner(int module_id, int level, const char *fmt, ...) { dav_log(module_id, fmt); }

void DlogWithKVInner(int module_id, int level, KeyValue *pst_kv_array, int kv_num, const char *fmt, ...) {
  dav_log(module_id, fmt);
}

int dlog_setlevel(int module_id, int level, int enable_event) {
  log_level = level;
  return log_level;
}

int dlog_getlevel(int module_id, int *enable_event) { return log_level; }

int CheckLogLevel(int moduleId, int log_level_check) { return log_level >= log_level_check; }

/**
 * @ingroup plog
 * @brief DlogReportInitialize: init log in service process before all device setting.
 * @return: 0: SUCCEED, others: FAILED
 */
int DlogReportInitialize() { return 0; }

/**
 * @ingroup plog
 * @brief DlogReportFinalize: release log resource in service process after all device reset.
 * @return: 0: SUCCEED, others: FAILED
 */
int DlogReportFinalize() { return 0; }
