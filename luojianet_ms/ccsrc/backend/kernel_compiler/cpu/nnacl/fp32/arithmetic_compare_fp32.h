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

#ifndef LUOJIANET_MS_NNACL_ARITHMETIC_COMPARE_H_
#define LUOJIANET_MS_NNACL_ARITHMETIC_COMPARE_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int ElementEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);

int ElementNotEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementNotEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);

int ElementLessFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementLessInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);

int ElementLessEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementLessEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);

int ElementGreaterFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementGreaterInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);

int ElementGreaterEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size);
int ElementGreaterEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size);
#ifdef __cplusplus
}
#endif

#endif  // LUOJIANET_MS_NNACL_ARITHMETIC_COMPARE_H_
