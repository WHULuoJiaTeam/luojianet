/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_

#include <limits.h>
#include <signal.h>

namespace mindspore {
namespace dataset {
// Register the custom signal handlers
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
extern void RegisterHandlers();

// A signal handler for SIGINT.  Drives interrupt to watchdog
extern void IntHandler(int sig_num,          // The signal that was raised
                       siginfo_t *sig_info,  // The siginfo structure.
                       void *context);       // context info
#endif
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_
