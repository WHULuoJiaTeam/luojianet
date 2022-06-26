/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_L2NORM_FP32_H_
#define MINDSPORE_NNACL_FP32_L2NORM_FP32_H_

#include "nnacl/l2_norm_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int CalcThreadSquareSum(const float *input_ptr, float *sum, int begin, int end);
int ThreadDivSqrtSum(const float *input_ptr, float *output_ptr, const L2NormParameter *param, const float sqrt_sum,
                     const int begin, const int end);
int ThreadTrailingAxis(const float *input_ptr, float *output_ptr, const L2NormParameter *param, const int begin,
                       const int end);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_L2NORM_FP32_H_
