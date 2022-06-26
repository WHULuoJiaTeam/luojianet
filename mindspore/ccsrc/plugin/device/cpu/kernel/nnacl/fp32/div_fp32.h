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
#ifndef MINDSPORE_NNACL_FP32_DIV_H_
#define MINDSPORE_NNACL_FP32_DIV_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/base/arithmetic_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int ElementDiv(const float *in0, const float *in1, float *out, int size);
int ElementDivRelu(const float *in0, const float *in1, float *out, int size);
int ElementDivRelu6(const float *in0, const float *in1, float *out, int size);
int ElementOptDiv(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptDivRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptDivRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptDivInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int BroadcastDiv(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_DIV_H_
