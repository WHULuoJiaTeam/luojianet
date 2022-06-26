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

#ifndef MINDSPORE_NNACL_SSE_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_NNACL_SSE_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <math.h>

#ifdef _MSC_VER
#include <immintrin.h>
#define MS_F32X4_GETI(src, i) src.m128_f32[i]
#else
#include <x86intrin.h>
#define MS_F32X4_GETI(src, i) src[i]
#endif

#define MS_FLOAT32X4 __m128
#define MS_INT32X4 __m128i
#define MS_MASK128_TYPE MS_FLOAT32X4
#define MS_LDQ_F32 _mm_loadu_ps
#define MS_LD128_F32 _mm_loadu_ps
#define MS_LDQ_EPI32(src) _mm_loadu_si128((__m128i const *)(src))
#define MS_LD128_EPI32(src) _mm_loadu_si128((__m128i const *)(src))
#define MS_ADDQ_F32 _mm_add_ps
#define MS_ADD128_F32 _mm_add_ps
#define MS_ADDQ_EPI32 _mm_add_epi32
#define MS_ADD128_EPI32 _mm_add_epi32
#define MS_MOVQ_F32 _mm_set1_ps
#define MS_MOV128_F32 _mm_set1_ps
#define MS_MOVQ_EPI32 _mm_set1_epi32
#define MS_MOV128_EPI32 _mm_set1_epi32
#define MS_MLAQ_F32(src1, src2, src3) _mm_add_ps(src1, _mm_mul_ps(src2, src3))
#define MS_STQ_F32 _mm_storeu_ps
#define MS_ST128_F32 _mm_storeu_ps
#define MS_STQ_EPI32(src1, src2) _mm_storeu_si128((__m128i *)(src1), src2)
#define MS_ST128_EPI32(src1, src2) _mm_storeu_si128((__m128i *)(src1), src2)
#define MS_SUBQ_F32 _mm_sub_ps
#define MS_SUB128_F32 _mm_sub_ps
#define MS_SUB128_EPI32 _mm_sub_epi32
#define MS_MAXQ_F32 _mm_max_ps
#define MS_MAXQ_EPI32 _mm_max_epi32
#define MS_MAX128_F32 _mm_max_ps
#define MS_MAX128_EPI32 _mm_max_epi32
#define MS_MINQ_F32 _mm_min_ps
#define MS_MINQ_EPI32 _mm_min_epi32
#define MS_SQRT128_F32 _mm_sqrt_ps
#define MS_RSQRT128_F32 _mm_rsqrt_ps
#define MS_LOG128_F32 _mm_log_ps
#define MS_COS128_F32 _mm_cos_ps
#define MS_SIN128_F32 _mm_sin_ps
#define MS_ERF128_F32 _mm_erf_ps
#define MS_ABS128_F32 _mm_abs_ps
#define MS_ROUND128_F32(src) _mm_round_ps(src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define MS_FLOOR128_F32 _mm_floor_ps
#define MS_CEIL128_F32 _mm_ceil_ps
#define MS_MULQ_F32(src1, src2) _mm_mul_ps(src1, src2)
#define MS_MULQ_EPI32(src1, src2) _mm_mullo_epi32(src1, src2)
#define MS_MIN128_F32 _mm_min_ps
#define MS_MIN128_EPI32 _mm_min_epi32
#define MS_MUL128_F32(src1, src2) _mm_mul_ps(src1, src2)
#define MS_MUL128_EPI32(src1, src2) _mm_mullo_epi32(src1, src2)
#define MS_DIVQ_F32(src1, src2) _mm_div_ps(src1, src2)
#define MS_DIV128_F32(src1, src2) _mm_div_ps(src1, src2)
#define MS_MULQ_N_F32(src1, src2) _mm_mul_ps(src1, _mm_set1_ps(src2))
#define MS_MULQ_N_EPI32(src1, src2) _mm_mullo_epi32(src1, _mm_set1_epi32(src2))
#define MS_DIVQ_N_F32(src1, src2) _mm_div_ps(src1, _mm_set1_ps(src2))
#define MS_SLLIQ_EPI32(src1, src2) _mm_slli_epi32(src1, src2)
#define MS_CVTQPS_EPI32(src) _mm_cvttps_epi32(src)  // truncate float to int
#define MS_CVTQEPI32_PS(src) _mm_cvtepi32_ps(src)
#define MS_CMPLEQ_F32(src1, src2) _mm_cmple_ps(src1, src2)
#define MS_CMPGTQ_F32(src1, src2) _mm_cmpgt_ps(src1, src2)
#define MS_CMPGTQ_EPI32(src1, src2) _mm_cmpgt_epi32(src1, src2)
#define MS_BLENDQ_F32(src1, src2, src3) _mm_blendv_ps(src1, src2, src3)
#define MS_BLENDQ_EPI32(src1, src2, src3) _mm_blendv_epi8(src1, src2, src3)
#define MS_CMPLE128_F32(src1, src2) _mm_cmple_ps(src1, src2)
#define MS_CMPGT128_F32(src1, src2) _mm_cmpgt_ps(src1, src2)
#define MS_CMPGT128_EPI32(src1, src2) _mm_cmpgt_epi32(src1, src2)
#define MS_BLEND128_F32(src1, src2, src3) _mm_blendv_ps(src1, src2, src3)
#define MS_BLEND128_EPI32(src1, src2, src3) _mm_blendv_epi8(src1, src2, src3)
#define MS_CAST_F32_S32(src) _mm_castsi128_ps(src)
#define MS_DIV128_EPI32(src1, src2) _mm_cvttps_epi32(MS_DIV128_F32(_mm_cvtepi32_ps(src1), _mm_cvtepi32_ps(src2)))

#ifdef ENABLE_AVX  // only enable sse, dont support fma instruction.
#define MS_FMADD128_F32(src1, src2, src3) _mm_fmadd_ps(src1, src2, src3)
#define MS_FMSUB128_F32(src1, src2, src3) _mm_fmsub_ps(src1, src2, src3)
#else
#define MS_FMADD128_F32(src1, src2, src3) _mm_add_ps(_mm_mul_ps(src1, src2), src3)
#define MS_FMSUB128_F32(src1, src2, src3) _mm_sub_ps(_mm_mul_ps(src1, src2), src3)
#endif

#define MS128_INT16_TO_FLOAT16(src) _mm_cvtepi16_ph(src)
#define MS128_FLOAT16_TO_INT16(src) _mm_cvttph_epi16(src)

#define MS128_INT32_TO_FLOAT16(src) _mm_cvtepi32_ph(src)
#define MS128_FLOAT16_TO_INT32(src) _mm_cvttph_epi32(src)

#define MS128_INT32_TO_FLOAT32(src) _mm_cvtepi32_ps(src)
#define MS128_FLOAT32_TO_INT32(src) _mm_cvttps_epi32(src)

#define MS128_INT64_TO_FLOAT32(src) _mm_cvtepi64_ps(src)
#define MS128_FLOAT32_TO_INT64(src) _mm_cvttps_epi64(src)

#define MS128_INT64_TO_FLOAT16(src) _mm_cvtepi64_ph(src)
#define MS128_FLOAT16_TO_INT64(src) _mm_cvttph_epi64(src)

#define MS128_INT32_TO_FLOAT64(src) _mm_cvtepi32_pd(src)
#define MS128_FLOAT64_TO_INT32(src) _mm_cvttpd_epi32(src)

#define MS128_INT64_TO_FLOAT64(src) _mm_cvtepi64_pd(src)
#define MS128_FLOAT64_TO_INT64(src) _mm_cvttpd_epi64(src)

#define MS128_INT16_TO_INT32(src) _mm128_cvtepi16_epi32(src)
#define MS128_INT16_TO_INT64(src) _mm128_cvtepi16_epi64(src)
#define MS128_INT32_TO_INT16(src) _mm128_cvtepi32_epi16(src)
#define MS128_INT32_TO_INT64(src) _mm128_cvtepi32_epi64(src)
#define MS128_INT64_TO_INT16(src) _mm128_cvtepi64_epi16(src)
#define MS128_INT64_TO_INT32(src) _mm128_cvtepi64_epi32(src)

static inline MS_FLOAT32X4 MS_SQRTFX4_F32(MS_FLOAT32X4 src) {
  MS_FLOAT32X4 dst;
  MS_F32X4_GETI(dst, 0) = sqrtf(MS_F32X4_GETI(src, 0));
  MS_F32X4_GETI(dst, 1) = sqrtf(MS_F32X4_GETI(src, 1));
  MS_F32X4_GETI(dst, 2) = sqrtf(MS_F32X4_GETI(src, 2));
  MS_F32X4_GETI(dst, 3) = sqrtf(MS_F32X4_GETI(src, 3));
  return dst;
}

static inline float MS_GET_MAX128_F32(__m128 src) {
  float result = MS_F32X4_GETI(src, 0);
  for (int i = 1; i < 4; i++) {  // sse block num : 4
    result = fmaxf(result, MS_F32X4_GETI(src, i));
  }
  return result;
}

static inline float MS_GET_SUM128_F32(__m128 src) {
  float result = MS_F32X4_GETI(src, 0);
  for (int i = 1; i < 4; i++) {  // sse block num : 4
    result = result + MS_F32X4_GETI(src, i);
  }
  return result;
}

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
  MS_FLOAT32X4 square = MS_MULQ_F32(src, src);
  MS_FLOAT32X4 a = MS_MULQ_F32(
    MS_ADDQ_F32(MS_MULQ_F32(MS_ADDQ_F32(MS_MULQ_F32(MS_ADDQ_F32(square, data0), square), data1), square), data2), src);
  MS_FLOAT32X4 b = MS_ADDQ_F32(
    MS_MULQ_F32(MS_ADDQ_F32(MS_MULQ_F32(MS_ADDQ_F32(MS_MULQ_F32(data3, square), data4), square), data5), square),
    data2);
  return MS_MINQ_F32(MS_MAXQ_F32(MS_DIVQ_F32(a, b), neg), pos);
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
#endif  // MINDSPORE_NNACL_SSE_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
