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
#ifndef LUOJIANET_MS_NNACL_RAGGED_RANGE_FP32_H_
#define LUOJIANET_MS_NNACL_RAGGED_RANGE_FP32_H_

#include <math.h>
#include "nnacl/ragged_range_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void RaggedRangeFp32(const float *starts, const float *limits, const float *deltas, int *splits, float *value,
                     const RaggedRangeParameter *param);
void RaggedRangeInt(const int *starts, const int *limits, const int *deltas, int *splits, int *value,
                    const RaggedRangeParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // LUOJIANET_MS_NNACL_RAGGED_RANGE_FP32_H_
