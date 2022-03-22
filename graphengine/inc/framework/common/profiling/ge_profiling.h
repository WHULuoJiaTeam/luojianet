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

#ifndef INC_FRAMEWORK_COMMON_GE_PROFILING_H_
#define INC_FRAMEWORK_COMMON_GE_PROFILING_H_

#include "external/ge/ge_api_error_codes.h"
#include "runtime/base.h"

///
/// @brief Output the profiling data of single operator in Pytorch, and does not support multithreading
/// @return Status result
///
GE_FUNC_VISIBILITY ge::Status ProfSetStepInfo(uint64_t index_id, uint16_t tag_id, rtStream_t stream);

GE_FUNC_VISIBILITY ge::Status ProfGetDeviceFormGraphId(uint32_t graph_id, uint32_t &device_id);

GE_FUNC_VISIBILITY void ProfSetGraphIdToDeviceMap(uint32_t graph_id, uint32_t &device_id);

#endif  // INC_FRAMEWORK_COMMON_GE_PROFILING_H_
