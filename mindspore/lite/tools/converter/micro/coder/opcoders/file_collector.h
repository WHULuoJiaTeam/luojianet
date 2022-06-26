/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_CODER_FILE_COLLECTOR_H_
#define MINDSPORE_LITE_MICRO_CODER_FILE_COLLECTOR_H_

#include <string>
#include <vector>
#include "tools/converter/micro/coder/context.h"

namespace mindspore::lite::micro {
void Collect(CoderContext *const ctx, const std::vector<std::string> &headers,
             const std::vector<std::string> &cFiles = {}, const std::vector<std::string> &asmFiles = {});
}
// namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_FILE_COLLECTOR_H_
