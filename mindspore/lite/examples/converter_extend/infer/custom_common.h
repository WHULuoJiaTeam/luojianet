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

#ifndef MINDSPORE_LITE_EXAMPLES_CONVERTER_EXTEND_INFER_CUSTOM_COMMON_H
#define MINDSPORE_LITE_EXAMPLES_CONVERTER_EXTEND_INFER_CUSTOM_COMMON_H

#include <vector>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/ms_tensor.h"

namespace mindspore {
namespace common {
// verify that the inputs' shape is inferred successfully when inferring current node.
Status CheckInputs(const std::vector<mindspore::MSTensor> &inputs);
}  // namespace common
}  // namespace mindspore
#endif  // MINDSPORE_LITE_EXAMPLES_CONVERTER_EXTEND_INFER_CUSTOM_COMMON_H
