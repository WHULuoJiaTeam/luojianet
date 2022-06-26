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
#ifndef MINDSPORE_NNACL_FP32_GRAD_ARITHMETIC_GRAD_H_
#define MINDSPORE_NNACL_FP32_GRAD_ARITHMETIC_GRAD_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void ElementDivNegSquare(const float *nom, const float *denom, float *output, int element_size);
void ElementMulAndDivNegSquare(const float *a, const float *b, const float *denom, float *output, int element_size);
int ElementAbsGrad(const float *in1, const float *in2, float *out, int element_size);
void MaximumByAxes(const float *input0, const float *input1, const float *dy, const int *input0_dims,
                   const int *input1_dims, const int *dy_dims, float *output0, float *output1, int num_dims);
void MinimumByAxes(const float *input0, const float *input1, const float *dy, const int *input0_dims,
                   const int *input1_dims, const int *dy_dims, float *output0, float *output1, int num_dims);
int ElementSqrtGrad(const float *in1, const float *in2, float *out, const int element_size);
int ElementRsqrtGrad(const float *in1, const float *in2, float *out, const int element_size);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_GRAD_ARITHMETIC_GRAD_H_
