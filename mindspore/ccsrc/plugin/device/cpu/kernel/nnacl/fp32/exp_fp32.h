/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_EXP_H_
#define MINDSPORE_NNACL_FP32_EXP_H_

#include "nnacl/op_base.h"
#include "nnacl/exp_parameter.h"
#ifdef ENABLE_AVX512
#include "nnacl/intrinsics/ms_simd_avx512_instructions.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ExpFp32(const float *src, float *dst, int num);
int ExpFusionFp32(const float *src, float *dst, const ExpParameter *param, int task_id);

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
static inline MS_FLOAT32X4 VexpFp32(MS_FLOAT32X4 input) {
  static MS_FLOAT32X4 param[] = {{0.693147f, 0.693147f, 0.693147f, 0.693147f},
                                 {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
                                 {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
                                 {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
                                 {0.5f, 0.5f, 0.5f, 0.5f},
                                 {1.0f, 1.0f, 1.0f, 1.0f}};
  MS_INT32X4 integer = MS_CVTQPS_EPI32(MS_DIVQ_F32(input, param[0]));
  MS_FLOAT32X4 decimal = MS_SUBQ_F32(input, MS_MULQ_F32(MS_CVTQEPI32_PS(integer), param[0]));
  MS_INT32X4 int_exp = MS_SLLIQ_EPI32(MS_ADDQ_EPI32(integer, MS_MOVQ_EPI32(127)), 23);
  MS_FLOAT32X4 tmp = MS_MULQ_F32(decimal, (MS_ADDQ_F32(param[2], MS_MULQ_F32(decimal, param[1]))));
  tmp = MS_MULQ_F32(decimal, MS_ADDQ_F32(param[4], MS_MULQ_F32(decimal, MS_ADDQ_F32(param[3], tmp))));
  MS_FLOAT32X4 decimal_exp = MS_ADDQ_F32(param[5], MS_MULQ_F32(decimal, MS_ADDQ_F32(param[5], tmp)));
  return MS_MULQ_F32(decimal_exp, MS_CAST_F32_S32(int_exp));
}

static inline void simd_exp128(MS_FLOAT32X4 input, float *dst) {
  static MS_FLOAT32X4 maxv = {88.0f, 88.0f, 88.0f, 88.0f};
  static MS_FLOAT32X4 minv = {-88.0f, -88.0f, -88.0f, -88.0f};
  input = MS_MAXQ_F32(minv, MS_MINQ_F32(input, maxv));
  MS_STQ_F32(dst, VexpFp32(input));
}

static inline MS_FLOAT32X4 simd_exp128_f32(MS_FLOAT32X4 input) {
  static MS_FLOAT32X4 maxv = {88.0f, 88.0f, 88.0f, 88.0f};
  static MS_FLOAT32X4 minv = {-88.0f, -88.0f, -88.0f, -88.0f};
  input = MS_MAXQ_F32(minv, MS_MINQ_F32(input, maxv));
  return VexpFp32(input);
}

#endif

#if defined(ENABLE_AVX512)
static inline MS_FLOAT32X16 simd_exp512_f32(MS_FLOAT32X16 input) {
  static MS_FLOAT32X16 maxv = {88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f,
                               88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f};
  static MS_FLOAT32X16 minv = {-88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f,
                               -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f};
  static MS_FLOAT32X16 param[] = {
    {0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f,
     0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f},
    {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120,
     1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
    {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24,
     1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
    {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6,
     1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
    {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
  input = MS_MAX512_F32(minv, MS_MIN512_F32(input, maxv));
  MS_INT32X16 integer = MS_CVT512PS_EPI32(MS_DIV512_F32(input, param[0]));
  MS_FLOAT32X16 decimal = MS_SUB512_F32(input, MS_MUL512_F32(MS_CVT512EPI32_PS(integer), param[0]));
  MS_INT32X16 int_exp = MS_SLLI512_EPI32(MS_ADD512_EPI32(integer, MS_MOV512_EPI32(127)), 23);
  MS_FLOAT32X16 tmp = MS_MUL512_F32(decimal, (MS_ADD512_F32(param[2], MS_MUL512_F32(decimal, param[1]))));
  tmp = MS_MUL512_F32(decimal, MS_ADD512_F32(param[4], MS_MUL512_F32(decimal, MS_ADD512_F32(param[3], tmp))));
  MS_FLOAT32X16 decimal_exp = MS_ADD512_F32(param[5], MS_MUL512_F32(decimal, MS_ADD512_F32(param[5], tmp)));
  return MS_MUL512_F32(decimal_exp, MS_CAST512_F32_S32(int_exp));
}

static inline void simd_exp512(MS_FLOAT32X16 input, float *dst) {
  MS_FLOAT32X16 res = simd_exp512_f32(input);
  MS_ST512_F32(dst, res);
}
#endif

#if defined(ENABLE_AVX)
static inline MS_FLOAT32X8 simd_exp256_f32(MS_FLOAT32X8 input) {
  static MS_FLOAT32X8 maxv = {88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f};
  static MS_FLOAT32X8 minv = {-88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f};
  static MS_FLOAT32X8 param[] = {
    {0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f},
    {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
    {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
    {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
    {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
  input = MS_MAX256_F32(minv, MS_MIN256_F32(input, maxv));
  MS_INT32X8 integer = MS_CVT256PS_EPI32(MS_DIV256_F32(input, param[0]));
  MS_FLOAT32X8 decimal = MS_SUB256_F32(input, MS_MUL256_F32(MS_CVT256EPI32_PS(integer), param[0]));
  MS_INT32X8 int_exp = MS_SLLI256_EPI32(MS_ADD256_EPI32(integer, MS_MOV256_EPI32(127)), 23);
  MS_FLOAT32X8 tmp = MS_MUL256_F32(decimal, (MS_ADD256_F32(param[2], MS_MUL256_F32(decimal, param[1]))));
  tmp = MS_MUL256_F32(decimal, MS_ADD256_F32(param[4], MS_MUL256_F32(decimal, MS_ADD256_F32(param[3], tmp))));
  MS_FLOAT32X8 decimal_exp = MS_ADD256_F32(param[5], MS_MUL256_F32(decimal, MS_ADD256_F32(param[5], tmp)));
  return MS_MUL256_F32(decimal_exp, MS_CAST256_F32_S32(int_exp));
}

static inline void simd_exp256(MS_FLOAT32X8 input, float *dst) {
  MS_FLOAT32X8 res = simd_exp256_f32(input);
  MS_ST256_F32(dst, res);
}
#endif

static inline float simd_exp32_f32(float data) {
  typedef union {
    float f;
    int i;
  } fi;
  static float param[] = {0.693147f, 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};  // Approximate calculation param
  data = MSMAX(-88.0f, MSMIN(88.0f, data));                                             // clamp(-88, 88)
  int integer = data / param[0];
  float decimal = data - integer * param[0];
  fi int_exp;
  int_exp.i = (integer + 127) << 23;  // Approximate calculation : (integer + 127) << 23
  // Approximate calculation
  const float decimal_exp =
    1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  return int_exp.f * decimal_exp;
}

static inline void simd_exp32(float src, float *dst) {
  typedef union {
    float f;
    int i;
  } fi;
  static float param[] = {0.693147f, 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};  // log(2.0f)
  src = MSMAX(-88.0f, MSMIN(88.0f, src));
  int integer = src / param[0];
  float decimal = src - integer * param[0];
  fi int_exp;
  int_exp.i = (integer + 127) << 23;
  const float decimal_exp =
    1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  *dst = int_exp.f * decimal_exp;
}
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_EXP_H_
