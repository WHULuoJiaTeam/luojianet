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
#ifndef MINDSPORE_NNACL_ARITHMETIC_H_
#define MINDSPORE_NNACL_ARITHMETIC_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/base/arithmetic_base.h"
#include "nnacl/errorcode.h"
#include "nnacl/fp32/add_fp32.h"
#include "nnacl/fp32/mul_fp32.h"
#include "nnacl/fp32/div_fp32.h"
#include "nnacl/fp32/sub_fp32.h"
#include "nnacl/fp32/squared_difference.h"

#ifdef __cplusplus
extern "C" {
#endif
void TileOneDimensionFp32(const float *inData, float *outData, int dim, size_t ndim, const int *inShape,
                          const int *inStrides, const int *outStrides, const int *multiple);
void TileDimensionsFp32(const float *data0, const float *data1, float *tile_data0, float *tile_data1,
                        ArithmeticParameter *param);
/* logical and */
int ElementLogicalAnd(const float *in0, const float *in1, float *out, int size);
int ElementOptLogicalAnd(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementLogicalAndInt(const int *in0, const int *in1, int *out, int size);
int ElementOptLogicalAndInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int ElementLogicalAndBool(const bool *in0, const bool *in1, bool *out, int size);
int ElementOptLogicalAndBool(const bool *in0, const bool *in1, bool *out, int size, const ArithmeticParameter *param);

/* logical or */
int ElementLogicalOr(const float *in0, const float *in1, float *out, int size);
int ElementOptLogicalOr(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementLogicalOrBool(const bool *in0, const bool *in1, bool *out, int size);
int ElementOptLogicalOrBool(const bool *in0, const bool *in1, bool *out, int size, const ArithmeticParameter *param);

/* max min */
int ElementMaximum(const float *in0, const float *in1, float *out, int size);
int ElementOptMaximum(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementMinimum(const float *in0, const float *in1, float *out, int size);
int ElementOptMinimum(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementMaximumInt(const int *in0, const int *in1, int *out, int size);
int ElementOptMaximumInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int ElementMinimumInt(const int *input0, const int *input1, int *output, int size);
int ElementOptMinimumInt(const int *input0, const int *input1, int *output, int size, const ArithmeticParameter *param);

/* floor div */
int ElementFloorDiv(const float *in0, const float *in1, float *out, int size);
int ElementOptFloorDiv(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementFloorDivInt(const int *in0, const int *in1, int *out, int size);
int ElementOptFloorDivInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);

/* floor mod */
int ElementFloorMod(const float *in0, const float *in1, float *out, int size);
int ElementOptFloorMod(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementFloorModInt(const int *in0, const int *in1, int *out, int size);
int ElementOptFloorModInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);

/* mod */
int ElementMod(const float *in0, const float *in1, float *out, int size);
int ElementOptMod(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementModInt(const int *in0, const int *in1, int *out, int size);
int ElementOptModInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_ARITHMETIC_H_
