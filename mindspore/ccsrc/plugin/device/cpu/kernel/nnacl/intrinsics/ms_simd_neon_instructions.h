/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_NEON_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_NNACL_NEON_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <math.h>
#include <float.h>

#include <arm_neon.h>

#define MS_F32X4_GETI(src, i) src[i]
#define MS_FLOAT32X4 float32x4_t
#define MS_INT32X4 int32x4_t
#define MS_UINT32X4 uint32x4_t
#define MS_MASK128_TYPE MS_UINT32X4
#define MS_LDQ_F32 vld1q_f32
#define MS_LD128_F32 vld1q_f32
#define MS_LDQ_EPI32 vld1q_s32
#define MS_LD128_EPI32 vld1q_s32
#define MS_ADDQ_F32 vaddq_f32
#define MS_ADD128_F32 vaddq_f32
#define MS_ADDQ_EPI32 vaddq_s32
#define MS_ADD128_EPI32 vaddq_s32
#define MS_MOVQ_F32 vmovq_n_f32
#define MS_MOV128_F32 vmovq_n_f32
#define MS_MOVQ_EPI32 vmovq_n_s32
#define MS_MOV128_EPI32 vmovq_n_s32
#define MS_SUBQ_F32 vsubq_f32
#define MS_SUB128_F32 vsubq_f32
#define MS_SUB128_EPI32 vsubq_s32
#define MS_MLAQ_F32(src1, src2, src3) vmlaq_f32(src1, src2, src3)
#define MS_STQ_F32 vst1q_f32
#define MS_ST128_F32 vst1q_f32
#define MS_STQ_EPI32 vst1q_s32
#define MS_ST128_EPI32 vst1q_s32
#define MS_MAXQ_F32 vmaxq_f32
#define MS_MAXQ_EPI32 vmaxq_s32
#define MS_MAX128_F32 vmaxq_f32
#define MS_MAX128_EPI32 vmaxq_s32
#define MS_MINQ_F32 vminq_f32
#define MS_MINQ_EPI32 vminq_s32
#define MS_MULQ_F32(src1, src2) vmulq_f32(src1, src2)
#define MS_MULQ_EPI32(src1, src2) vmulq_s32(src1, src2)
#define MS_MIN128_F32 vminq_f32
#define MS_MIN128_EPI32 vminq_s32
#define MS_MUL128_F32(src1, src2) vmulq_f32(src1, src2)
#define MS_MUL128_EPI32(src1, src2) vmulq_s32(src1, src2)
#define MS_FMADD128_F32(src1, src2, src3) vmlaq_f32(src3, src1, src2)
#define MS_FMSUB128_EPI32(src1, src2, src3) vmlsq_s32(src3, src1, src2)
#ifdef ENABLE_ARM64
#define MS_DIVQ_F32(src1, src2) vdivq_f32(src1, src2)
#define MS_DIV128_F32(src1, src2) vdivq_f32(src1, src2)
#else
static inline float32x4_t vrecp(float32x4_t v) {
  float32x4_t r = vrecpeq_f32(v);
  r = vmulq_f32(vrecpsq_f32(v, r), r);
  r = vmulq_f32(vrecpsq_f32(v, r), r);
  return r;
}
#define MS_DIVQ_F32(src1, src2) vmulq_f32(src1, vrecp(src2))
#define MS_DIV128_F32(src1, src2) vmulq_f32(src1, vrecp(src2))
#endif
#define MS_MULQ_N_F32(src1, src2) vmulq_n_f32(src1, src2)
#define MS_MULQ_N_EPI32(src1, src2) vmulq_n_s32(src1, src2)
#define MS_DIVQ_N_F32(src1, src2) vdivq_n_f32(src1, src2)
#define MS_SLLIQ_EPI32(src1, src2) vshlq_s32(src1, vmovq_n_s32(src2))
#define MS_CVTQPS_EPI32(src) vcvtq_s32_f32(src)
#define MS_CVTQEPI32_PS(src) vcvtq_f32_s32(src)
#define MS_CMPLEQ_F32(src1, src2) vcleq_f32(src1, src2)
#define MS_CMPGTQ_F32(src1, src2) vcgtq_f32(src1, src2)
#define MS_CMPGTQ_EPI32(src1, src2) vcgtq_s32(src1, src2)
#define MS_CMPLE128_F32(src1, src2) vcleq_f32(src1, src2)
#define MS_CMPGT128_F32(src1, src2) vcgtq_f32(src1, src2)
#define MS_CMPGT128_EPI32(src1, src2) vcgtq_s32(src1, src2)
// Note: Compared with X86, the vbslq_f32 parameters are the opposite with _mm_blendv_f32
#define MS_BLENDQ_F32(src1, src2, src3) vbslq_f32(src3, src2, src1)
#define MS_BLENDQ_EPI32(src1, src2, src3) vbslq_s32(src3, src2, src1)
#define MS_BLEND128_F32(src1, src2, src3) vbslq_f32(src3, src2, src1)
#define MS_BLEND128_EPI32(src1, src2, src3) vbslq_s32(src3, src2, src1)
#define MS_CAST_F32_S32(src) vreinterpretq_f32_s32(src)

#ifdef ENABLE_ARM64
#define MS_GET_MAX128_F32 vmaxvq_f32
#else
static inline float MS_GET_MAX128_F32(MS_FLOAT32X4 src) {
  float result = MS_F32X4_GETI(src, 0);
  for (int i = 1; i < 4; i++) {  // neon block num : 4
    result = fmaxf(result, MS_F32X4_GETI(src, i));
  }
  return result;
}
#endif

static inline float MS_GET_SUM128_F32(MS_FLOAT32X4 src) {
  float result = MS_F32X4_GETI(src, 0);
  for (int i = 1; i < 4; i++) {  // neon block num : 4
    result = result + MS_F32X4_GETI(src, i);
  }
  return result;
}

static inline int32x4_t MS_DIV128_EPI32(int32x4_t src1, int32x4_t src2) {
  int32x4_t result;
  result[0] = src1[0] / src2[0];  // C0 : 0
  result[1] = src1[1] / src2[1];  // C1 : 1
  result[2] = src1[2] / src2[2];  // C2 : 2
  result[3] = src1[3] / src2[3];  // C3 : 3
  return result;
}

#define MS128_INT32_TO_FLOAT32(src) vcvtq_f32_s32(src)
#define MS128_FLOAT32_TO_INT32(src) vcvtq_s32_f32(src)

static inline MS_FLOAT32X4 MS_SQRTFX4_F32(MS_FLOAT32X4 src) {
  MS_FLOAT32X4 dst;
  MS_F32X4_GETI(dst, 0) = sqrtf(MS_F32X4_GETI(src, 0));
  MS_F32X4_GETI(dst, 1) = sqrtf(MS_F32X4_GETI(src, 1));
  MS_F32X4_GETI(dst, 2) = sqrtf(MS_F32X4_GETI(src, 2));
  MS_F32X4_GETI(dst, 3) = sqrtf(MS_F32X4_GETI(src, 3));
  return dst;
}

static inline MS_FLOAT32X4 MS_SQRT128_F32(MS_FLOAT32X4 src) {
  MS_FLOAT32X4 dst;
  MS_F32X4_GETI(dst, 0) = sqrtf(MS_F32X4_GETI(src, 0));
  MS_F32X4_GETI(dst, 1) = sqrtf(MS_F32X4_GETI(src, 1));
  MS_F32X4_GETI(dst, 2) = sqrtf(MS_F32X4_GETI(src, 2));
  MS_F32X4_GETI(dst, 3) = sqrtf(MS_F32X4_GETI(src, 3));
  return dst;
}
#define MS_RSQRT128_F32 vrsqrteq_f32

#define LOAD128X8_F32(src, input_ptr, num)               \
  MS_FLOAT32X4 src##1 = MS_LDQ_F32(input_ptr + 0 * num); \
  MS_FLOAT32X4 src##2 = MS_LDQ_F32(input_ptr + 1 * num); \
  MS_FLOAT32X4 src##3 = MS_LDQ_F32(input_ptr + 2 * num); \
  MS_FLOAT32X4 src##4 = MS_LDQ_F32(input_ptr + 3 * num); \
  MS_FLOAT32X4 src##5 = MS_LDQ_F32(input_ptr + 4 * num); \
  MS_FLOAT32X4 src##6 = MS_LDQ_F32(input_ptr + 5 * num); \
  MS_FLOAT32X4 src##7 = MS_LDQ_F32(input_ptr + 6 * num); \
  MS_FLOAT32X4 src##8 = MS_LDQ_F32(input_ptr + 7 * num);

#define STORE128X8_F32(output_ptr, num, dst) \
  MS_STQ_F32(output_ptr + 0 * num, dst##1);  \
  MS_STQ_F32(output_ptr + 1 * num, dst##2);  \
  MS_STQ_F32(output_ptr + 2 * num, dst##3);  \
  MS_STQ_F32(output_ptr + 3 * num, dst##4);  \
  MS_STQ_F32(output_ptr + 4 * num, dst##5);  \
  MS_STQ_F32(output_ptr + 5 * num, dst##6);  \
  MS_STQ_F32(output_ptr + 6 * num, dst##7);  \
  MS_STQ_F32(output_ptr + 7 * num, dst##8);

static inline MS_FLOAT32X4 MS_TANHX4_F32(MS_FLOAT32X4 src) {
  static const MS_FLOAT32X4 data0 = {378.0f, 378.0f, 378.0f, 378.0f};
  static const MS_FLOAT32X4 data1 = {17325.0f, 17325.0f, 17325.0f, 17325.0f};
  static const MS_FLOAT32X4 data2 = {135135.0f, 135135.0f, 135135.0f, 135135.0f};
  static const MS_FLOAT32X4 data3 = {28.0f, 28.0f, 28.0f, 28.0f};
  static const MS_FLOAT32X4 data4 = {3150.0f, 3150.0f, 3150.0f, 3150.0f};
  static const MS_FLOAT32X4 data5 = {62370.0f, 62370.0f, 62370.0f, 62370.0f};
  static const MS_FLOAT32X4 neg = {-1.0f, -1.0f, -1.0f, -1.0f};
  static const MS_FLOAT32X4 pos = {1.0f, 1.0f, 1.0f, 1.0f};
  static const MS_FLOAT32X4 up_limit = {5.0f, 5.0f, 5.0f, 5.0f};
  static const MS_FLOAT32X4 down_limit = {-5.0f, -5.0f, -5.0f, -5.0f};

  MS_UINT32X4 up_mask = MS_CMPGTQ_F32(src, up_limit);
  MS_UINT32X4 down_mask = MS_CMPGTQ_F32(down_limit, src);

  MS_FLOAT32X4 square = MS_MULQ_F32(src, src);
  MS_FLOAT32X4 a = MS_MULQ_F32(
    MS_ADDQ_F32(MS_MULQ_F32(MS_ADDQ_F32(MS_MULQ_F32(MS_ADDQ_F32(square, data0), square), data1), square), data2), src);
  MS_FLOAT32X4 b = MS_ADDQ_F32(
    MS_MULQ_F32(MS_ADDQ_F32(MS_MULQ_F32(MS_ADDQ_F32(MS_MULQ_F32(data3, square), data4), square), data5), square),
    data2);

  MS_FLOAT32X4 tanh_value = MS_DIVQ_F32(a, b);
  MS_FLOAT32X4 res = MS_BLENDQ_F32(tanh_value, pos, up_mask);
  res = MS_BLENDQ_F32(res, neg, down_mask);
  return res;
}

static inline MS_FLOAT32X4 MS_ERFX4_F32(MS_FLOAT32X4 src) {
  MS_FLOAT32X4 dst;
  MS_F32X4_GETI(dst, 0) = erff(MS_F32X4_GETI(src, 0));
  MS_F32X4_GETI(dst, 1) = erff(MS_F32X4_GETI(src, 1));
  MS_F32X4_GETI(dst, 2) = erff(MS_F32X4_GETI(src, 2));
  MS_F32X4_GETI(dst, 3) = erff(MS_F32X4_GETI(src, 3));
  return dst;
}

#define MS_FMADD128X8_F32(src, weight, dst)     \
  dst##1 = MS_MLAQ_F32(src##1, weight, dst##1); \
  dst##2 = MS_MLAQ_F32(src##2, weight, dst##2); \
  dst##3 = MS_MLAQ_F32(src##3, weight, dst##3); \
  dst##4 = MS_MLAQ_F32(src##4, weight, dst##4); \
  dst##5 = MS_MLAQ_F32(src##5, weight, dst##5); \
  dst##6 = MS_MLAQ_F32(src##6, weight, dst##6); \
  dst##7 = MS_MLAQ_F32(src##7, weight, dst##7); \
  dst##8 = MS_MLAQ_F32(src##8, weight, dst##8);

#define MS_LOAD128X4_F32(src, input_ptr, num)            \
  MS_FLOAT32X4 src##1 = MS_LDQ_F32(input_ptr + 0 * num); \
  MS_FLOAT32X4 src##2 = MS_LDQ_F32(input_ptr + 1 * num); \
  MS_FLOAT32X4 src##3 = MS_LDQ_F32(input_ptr + 2 * num); \
  MS_FLOAT32X4 src##4 = MS_LDQ_F32(input_ptr + 3 * num);

#define MS_FMADD128X4_F32(src, weight, dst)     \
  dst##1 = MS_MLAQ_F32(src##1, weight, dst##1); \
  dst##2 = MS_MLAQ_F32(src##2, weight, dst##2); \
  dst##3 = MS_MLAQ_F32(src##3, weight, dst##3); \
  dst##4 = MS_MLAQ_F32(src##4, weight, dst##4);

#define MS_LOAD128X8_F32(src, input_ptr, num)            \
  MS_FLOAT32X4 src##1 = MS_LDQ_F32(input_ptr + 0 * num); \
  MS_FLOAT32X4 src##2 = MS_LDQ_F32(input_ptr + 1 * num); \
  MS_FLOAT32X4 src##3 = MS_LDQ_F32(input_ptr + 2 * num); \
  MS_FLOAT32X4 src##4 = MS_LDQ_F32(input_ptr + 3 * num); \
  MS_FLOAT32X4 src##5 = MS_LDQ_F32(input_ptr + 4 * num); \
  MS_FLOAT32X4 src##6 = MS_LDQ_F32(input_ptr + 5 * num); \
  MS_FLOAT32X4 src##7 = MS_LDQ_F32(input_ptr + 6 * num); \
  MS_FLOAT32X4 src##8 = MS_LDQ_F32(input_ptr + 7 * num);

#define MS_SET_ZERO128X8_F32(dst)          \
  MS_FLOAT32X4 dst##1 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##2 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##3 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##4 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##5 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##6 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##7 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##8 = MS_MOVQ_F32(0.0f);

#define MS_SET_ZERO128X4_F32(dst)          \
  MS_FLOAT32X4 dst##1 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##2 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##3 = MS_MOVQ_F32(0.0f); \
  MS_FLOAT32X4 dst##4 = MS_MOVQ_F32(0.0f);
#endif  // MINDSPORE_NNACL_NEON_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
