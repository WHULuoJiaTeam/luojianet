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
#ifndef MINDSPORE_NNACL_FP16_ARITHMETIC_SELF_FP16_H_
#define MINDSPORE_NNACL_FP16_ARITHMETIC_SELF_FP16_H_

#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int ElementAbsFp16(const float16_t *input, float16_t *output, int element_size);

int ElementCosFp16(const float16_t *input, float16_t *output, int element_size);

int ElementLogFp16(const float16_t *input, float16_t *output, int element_size);

int ElementSquareFp16(const float16_t *input, float16_t *output, int element_size);

int ElementSqrtFp16(const float16_t *input, float16_t *output, int element_size);

int ElementRsqrtFp16(const float16_t *input, float16_t *output, int element_size);

int ElementSinFp16(const float16_t *input, float16_t *output, int element_size);

int ElementLogicalNotFp16(const float16_t *input, float16_t *output, int element_size);

int ElementRoundFp16(const float16_t *input, float16_t *output, int element_size);

int ElementFloorFp16(const float16_t *input, float16_t *output, int element_size);

int ElementCeilFp16(const float16_t *input, float16_t *output, int number);

int ElementNegativeFp16(const float16_t *input, float16_t *output, int element_size);

int ElementReciprocalFp16(const float16_t *input, float16_t *output, int element_size);

int ElementErfFp16(const float16_t *input, float16_t *output, int element_size);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_ARITHMETIC_SELF_FP16_H_
