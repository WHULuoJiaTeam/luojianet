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

#ifndef MINDSPORE_NNACL_FP32_MATMUL_H_
#define MINDSPORE_NNACL_FP32_MATMUL_H_

#include <float.h>
#include <string.h>
#include "nnacl/errorcode.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/op_base.h"

#define ADD_BIAS(value, bias, c) \
  if (bias != NULL) value = value + bias[c];

#define DO_RELU(value, act_type) \
  if (act_type == ActType_Relu) value = MSMAX(0.0f, value);

#define DO_RELU6(value, act_type)                            \
  if (act_type == ActType_Relu6) value = MSMIN(6.0f, value); \
  if (act_type == ActType_Relu6) value = MSMAX(0.0f, value);

#ifdef __cplusplus
extern "C" {
#endif
void MatMulOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row,
               int col, size_t stride, int out_type);
void MatVecMulFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);
void MatVecMulFp32Block8(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);
void MatVecMulFp32Block4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);

void RowMajor2ColMajor(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2RowMajor(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row4Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row6Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row8Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row12Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row16Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row32Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row64Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col4Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col6Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col8Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col12Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col16Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col32Major(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col64Major(const float *src_ptr, float *dst_ptr, int row, int col);

#ifdef ENABLE_ARM64
void MatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, size_t stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, size_t stride, size_t write_mode);
void MatmulFloatNeon64OptRow8(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                              int row, int col, size_t stride, size_t write_mode);
void MatmulFloatNeon64OptRow4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                              int row, int col, size_t stride, size_t write_mode);
void MatmulFloatNeon64OptRow12(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                               int row, int col, size_t stride, size_t write_mode);
void MatVecMulFp32Neon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col,
                         int align_col);

#elif defined(ENABLE_ARM32)
void MatmulFloatNeon32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, int stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon32Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, int stride, int write_mode);
void MatmulFloatNeon32Opt12x4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                              int row, int col, int stride, int write_mode);

#elif defined(ENABLE_AVX)
typedef void (*DeconvAvxKernel)(const float *src, const float *weight, float *dst, int col, int row, int depth,
                                int stride);
void DeconvMatmulAvx(const float *a, const float *b, float *c, int depth, int row, int col, int kernel_plane);
void MatmulFloatAvxOpt(const float *a, const float *b, float *c, const float *bias, size_t act_type, size_t depth,
                       size_t row, size_t col, size_t stride, size_t write_mode);
typedef void (*MatVecMulKernel)(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                                size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMulAvxFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                      int col_align);
void MatMulAvxFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                   int col_align, int row);
void MatVecMul1x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMul1x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMul1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatVecMul1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                        size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul3x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                      size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul4x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                      size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul6x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                      size_t row_block, size_t col_block, size_t col_algin, size_t deep);
void MatMul8x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                     size_t row_block, size_t col_block, size_t col_algin, size_t deep);
#ifdef ENABLE_DEBUG
void DeconvColXRowAvxKernel(const float *src, const float *weight, float *dst, int col, int row, int depth, int stride);

void MatVecMulRowxColKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t row_block, size_t col_block, size_t col_algin, size_t deep);
#endif

#elif defined(ENABLE_SSE)
void DeconvMatmulFloatSse(const float *a, const float *b, float *c, int depth, int row, int col);
void MatmulFloatSse64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                         int col, int stride, int write_mode);
#endif

void MatMul12x8(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
                int col, int stride, int out_type);

void GemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int row, int deep);

void GemmIsNotPackOptimize(const float *a, const float *b, float *c, const float *bias, int m, int k);

#ifdef ENABLE_ARM64
void GemmIsNotPackByRow(const float *a, const float *b, float *c, const float *bias, int start_row, int end_row,
                        int deep);
#endif
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP32_MATMUL_H_
