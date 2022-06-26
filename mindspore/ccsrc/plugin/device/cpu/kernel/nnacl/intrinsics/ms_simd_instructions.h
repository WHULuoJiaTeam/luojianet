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

#ifndef MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <math.h>

#ifdef ENABLE_AVX512
#include "nnacl/intrinsics/ms_simd_avx512_instructions.h"
#endif

#ifdef ENABLE_AVX
#include "nnacl/intrinsics/ms_simd_avx_instructions.h"
#endif

#ifdef ENABLE_SSE
#include "nnacl/intrinsics/ms_simd_sse_instructions.h"
#endif

#ifdef ENABLE_ARM
#include "nnacl/intrinsics/ms_simd_neon_instructions.h"
#endif

#define SIMD512_BLOCK16 32  // SIMD : 512 = 16 x 32
#define SIMD256_BLOCK16 16  // SIMD : 256 = 16 x 16
#define SIMD128_BLOCK16 8   // SIMD : 128 = 16 x 8

#define SIMD512_BLOCK32 16  // SIMD : 512 = 32 x 16
#define SIMD256_BLOCK32 8   // SIMD : 256 = 32 x 8
#define SIMD128_BLOCK32 4   // SIMD : 128 = 32 x 4

#define SIMD512_BLOCK64 8  // SIMD : 512 = 64 x 8
#define SIMD256_BLOCK64 4  // SIMD : 256 = 64 x 4
#define SIMD128_BLOCK64 2  // SIMD : 128 = 64 x 2

#define MS_EXPAND(...) __VA_ARGS__

// Scaler
#define MS_FLOAT32X1 float
#define MS_INT32X1 int
#define MS_MOV32_F32(value) (value)
#define MS_MOV32_EPI32(value) (value)
#define MS_LD32_F32(address) (*(address))
#define MS_LD32_EPI32(address) (*(address))
#define MS_ST32_F32(address, value) (*(address) = (value))
#define MS_ST32_EPI32(address, value) (*(address) = (value))
#define MS_ADD32_F32(value1, value2) ((value1) + (value2))
#define MS_ADD32_EPI32(value1, value2) ((value1) + (value2))
#define MS_SUB32_F32(value1, value2) ((value1) - (value2))
#define MS_SUB32_EPI32(value1, value2) ((value1) - (value2))
#define MS_MUL32_F32(value1, value2) ((value1) * (value2))
#define MS_MUL32_EPI32(value1, value2) ((value1) * (value2))
#define MS_DIV32_F32(value1, value2) ((value1) / (value2))
#define MS_DIV32_EPI32(value1, value2) ((value1) / (value2))
#define MS_MIN32_F32(value1, value2) (fmin((value1), (value2)))
#define MS_MIN32_EPI32(value1, value2) ((value1) < (value2) ? (value1) : (value2))
#define MS_MAX32_F32(value1, value2) (fmax((value1), (value2)))
#define MS_MAX32_EPI32(value1, value2) ((value1) > (value2) ? (value1) : (value2))

// define (float/int) data
#define MS_FLOAT_32xN(byte_num) MS_FLOAT32##X##byte_num
#define MS_INT_32xN(byte_num) MS_INT32##X##byte_num

// move (float/int) data
#define MS_MOVN_F32(byte_num, ...) MS_EXPAND(MS_MOV##byte_num##_F32(__VA_ARGS__))
#define MS_MOVN_EPI32(byte_num, ...) MS_EXPAND(MS_MOV##byte_num##_EPI32(__VA_ARGS__))

// load (float/int) data
#define MS_LD_F32(bit_num, ...) MS_EXPAND(MS_LD##bit_num##_F32(__VA_ARGS__))
#define MS_LD_EPI32(bit_num, ...) MS_EXPAND(MS_LD##bit_num##_EPI32(__VA_ARGS__))

// load 4 (float/int) data
#define MS_LDX4_F32(bit_num, ...) MS_EXPAND(MS_LOAD##bit_num##X4_F32(__VA_ARGS__))
#define MS_LDX4_EPI32(bit_num, ...) MS_EXPAND(MS_LOAD##bit_num##X4_EPI32(__VA_ARGS__))

// stored (float/int) data
#define MS_ST_F32(bit_num, ...) MS_EXPAND(MS_ST##bit_num##_F32(__VA_ARGS__))
#define MS_ST_EPI32(bit_num, ...) MS_EXPAND(MS_ST##bit_num##_EPI32(__VA_ARGS__))

// add (float/int) op
#define MS_ADD_F32(bit_num, ...) MS_EXPAND(MS_ADD##bit_num##_F32(__VA_ARGS__))
#define MS_ADD_EPI32(bit_num, ...) MS_EXPAND(MS_ADD##bit_num##_EPI32(__VA_ARGS__))
#define MS_ADD_N_F32(bit_num, val1, val2) MS_EXPAND(MS_ADD##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_ADD_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_ADD##bit_num##_EPI32(val1, MS_MOV##bit_num##_F32(val2)))

// sub (float/int) op
#define MS_SUB_F32(bit_num, ...) MS_EXPAND(MS_SUB##bit_num##_F32(__VA_ARGS__))
#define MS_SUB_EPI32(bit_num, ...) MS_EXPAND(MS_SUB##bit_num##_EPI32(__VA_ARGS__))
#define MS_SUB_N_F32(bit_num, val1, val2) MS_EXPAND(MS_SUB##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_SUB_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_SUB##bit_num##_EPI32(val1, MS_MOV##bit_num##_F32(val2)))

// div (float/int) op
#define MS_DIV_F32(bit_num, ...) MS_EXPAND(MS_DIV##bit_num##_F32(__VA_ARGS__))
#define MS_DIV_EPI32(bit_num, ...) MS_EXPAND(MS_DIV##bit_num##_EPI32(__VA_ARGS__))
#define MS_DIV_N_F32(bit_num, val1, val2) MS_EXPAND(MS_DIV##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_DIV_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_DIV##bit_num##_EPI32(val1, MS_MOV##bit_num##_EPI32(val2)))

// sqrt (float) op
#define MS_SQRT_F32(bit_num, ...) MS_EXPAND(MS_SQRT##bit_num##_F32(__VA_ARGS__))

// rsqrt (float) op
#define MS_RSQRT_F32(bit_num, ...) MS_EXPAND(MS_RSQRT##bit_num##_F32(__VA_ARGS__))

// log (float) op
#define MS_LOG_F32(bit_num, ...) MS_EXPAND(MS_LOG##bit_num##_F32(__VA_ARGS__))

// cos (float) op
#define MS_COS_F32(bit_num, ...) MS_EXPAND(MS_COS##bit_num##_F32(__VA_ARGS__))

// sin (float) op
#define MS_SIN_F32(bit_num, ...) MS_EXPAND(MS_SIN##bit_num##_F32(__VA_ARGS__))

// erf (float) op
#define MS_ERF_F32(bit_num, ...) MS_EXPAND(MS_ERF##bit_num##_F32(__VA_ARGS__))

// log (float) op
#define MS_ABS_F32(bit_num, ...) MS_EXPAND(MS_ABS##bit_num##_F32(__VA_ARGS__))

// round (float) op
#define MS_ROUND_F32(bit_num, ...) MS_EXPAND(MS_ROUND##bit_num##_F32(__VA_ARGS__))

// ceil (float) op
#define MS_CEIL_F32(bit_num, ...) MS_EXPAND(MS_CEIL##bit_num##_F32(__VA_ARGS__))

// floor (float) op
#define MS_FLOOR_F32(bit_num, ...) MS_EXPAND(MS_FLOOR##bit_num##_F32(__VA_ARGS__))

// min (float/int) op
#define MS_MIN_F32(bit_num, ...) MS_EXPAND(MS_MIN##bit_num##_F32(__VA_ARGS__))
#define MS_MIN_EPI32(bit_num, ...) MS_EXPAND(MS_MIN##bit_num##_EPI32(__VA_ARGS__))
#define MS_MIN_N_F32(bit_num, val, n) MS_MIN_F32(bit_num, val, MS_MOVN_F32(bit_num, n))
#define MS_MIN_N_EPI32(bit_num, val, n) MS_MIN_EPI32(bit_num, val, MS_MOVN_EPI32(bit_num, n))

// max (float/int) op
#define MS_MAX_F32(bit_num, ...) MS_EXPAND(MS_MAX##bit_num##_F32(__VA_ARGS__))
#define MS_MAX_EPI32(bit_num, ...) MS_EXPAND(MS_MAX##bit_num##_EPI32(__VA_ARGS__))

// get max (float/int) op
#define MS_GET_MAX_F32(bit_num, ...) MS_EXPAND(MS_GET_MAX##bit_num##_F32(__VA_ARGS__))
#define MS_GET_MAX_EPI32(bit_num, ...) MS_EXPAND(MS_GET_MAX##bit_num##_EPI32(__VA_ARGS__))

// get max (float/int) op
#define MS_GET_SUM_F32(bit_num, ...) MS_EXPAND(MS_GET_SUM##bit_num##_F32(__VA_ARGS__))

// max n (float/int) op
#define MS_MAX_N_F32(bit_num, val, n) MS_MAX_F32(bit_num, val, MS_MOVN_F32(bit_num, n))
#define MS_MAX_N_EPI32(bit_num, val, n) MS_MAX_EPI32(bit_num, val, MS_MOVN_EPI32(bit_num, n))
#define MS_CLAMP_F32(bit_num, val, min_val, max_val) MS_MIN_F32(bit_num, MS_MAX_F32(bit_num, val, min_val), max_val)
#define MS_CLAMP_EPI32(bit_num, val, min_val, max_val) \
  MS_MIN_EPI32(bit_num, MS_MAX_EPI32(bit_num, val, min_val), max_val)

// clamp n (float/int) op
#define MS_CLAMP_N_F32(bit_num, val, min_val, max_val) \
  MS_MIN_F32(bit_num, MS_MAX_F32(bit_num, val, MS_MOV##bit_num##_F32(min_val)), MS_MOV##bit_num##_F32(max_val))
#define MS_CLAMP_N_EPI32(bit_num, val, min_val, max_val) \
  MS_MIN_EPI32(bit_num, MS_MAX_EPI32(bit_num, val, MS_MOV##bit_num##_EPI32(min_val)), MS_MOV##bit_num##_EPI32(max_val))

// mul (float/int) op
#define MS_MUL_F32(bit_num, ...) MS_EXPAND(MS_MUL##bit_num##_F32(__VA_ARGS__))
#define MS_MUL_EPI32(bit_num, ...) MS_EXPAND(MS_MUL##bit_num##_EPI32(__VA_ARGS__))
#define MS_MUL_N_F32(bit_num, val1, val2) MS_EXPAND(MS_MUL##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_MUL_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_MUL##bit_num##_EPI32(val1, MS_MOV##bit_num##_EPI32(val2)))

// fma (float/int) op
#define MS_FMADD_F32(bit_num, ...) MS_EXPAND(MS_FMADD##bit_num##_F32(__VA_ARGS__))
#define MS_FMADD_N_F32(bit_num, val1, val2) MS_EXPAND(MS_FMADD##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))

// fms (float/int) op
#define MS_FMSUB_F32(bit_num, ...) MS_EXPAND(MS_FMSUB##bit_num##_F32(__VA_ARGS__))
#define MS_FMSUB_N_F32(bit_num, val1, val2) MS_EXPAND(MS_FMSUB##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))

// square (float/int) op
#define MS_MUL_SQUARE_F32(bit_num, val) MS_EXPAND((MS_MUL##bit_num##_F32(val, val)))
#define MS_MUL_SQUARE_EPI32(bit_num, val) MS_EXPAND((MS_MUL##bit_num##_EPI32(val, val)))

// exp (float) op
#define MS_EXP_ST_F32(bit_num, ...) MS_EXPAND((simd_exp##bit_num(__VA_ARGS__)))
#define MS_EXP_F32(bit_num, ...) MS_EXPAND((simd_exp##bit_num##_f32(__VA_ARGS__)))

#define MS_CMPLE_F32(bit_num, ...) MS_EXPAND((MS_CMPLE##bit_num##_F32(__VA_ARGS__)))
#define MS_CMPGT_F32(bit_num, ...) MS_EXPAND((MS_CMPGT##bit_num##_F32(__VA_ARGS__)))
#define MS_BLEND_F32(bit_num, ...) MS_EXPAND((MS_BLEND##bit_num##_F32(__VA_ARGS__)))

#define MS_INT16_TO_FLOAT16(bit_num, ...) MS_EXPAND((MS##bit_num##_INT16_TO_FLOAT16(__VA_ARGS__)))
#define MS_FLOAT16_TO_INT16(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT16_TO_INT16(__VA_ARGS__)))

#define MS_INT32_TO_FLOAT16(bit_num, ...) MS_EXPAND((MS##bit_num##_INT32_TO_FLOAT16(__VA_ARGS__)))
#define MS_FLOAT16_TO_INT32(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT16_TO_INT32(__VA_ARGS__)))

#define MS_INT32_TO_FLOAT32(bit_num, ...) MS_EXPAND((MS##bit_num##_INT32_TO_FLOAT32(__VA_ARGS__)))
#define MS_FLOAT32_TO_INT32(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT32_TO_INT32(__VA_ARGS__)))

#define MS_INT64_TO_FLOAT32(bit_num, ...) MS_EXPAND((MS##bit_num##_INT64_TO_FLOAT32(__VA_ARGS__)))
#define MS_FLOAT32_TO_INT64(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT32_TO_INT64(__VA_ARGS__)))

#define MS_INT64_TO_FLOAT16(bit_num, ...) MS_EXPAND((MS##bit_num##_INT64_TO_FLOAT16(__VA_ARGS__)))
#define MS_FLOAT16_TO_INT64(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT16_TO_INT64(__VA_ARGS__)))

#define MS_INT32_TO_FLOAT64(bit_num, ...) MS_EXPAND((MS##bit_num##_INT32_TO_FLOAT64(__VA_ARGS__)))
#define MS_FLOAT64_TO_INT32(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT64_TO_INT32(__VA_ARGS__)))

#define MS_INT64_TO_FLOAT64(bit_num, ...) MS_EXPAND((MS##bit_num##_INT64_TO_FLOAT64(__VA_ARGS__)))
#define MS_FLOAT64_TO_INT64(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT64_TO_INT64(__VA_ARGS__)))

// enable avx512
#if defined(ENABLE_AVX512)
#define MS_SIMD_RUN_AVX512(function, ...) MS_EXPAND(function(512, 16, __VA_ARGS__))
#else
#define MS_SIMD_RUN_AVX512(function, ...)
#endif

// enable avx256
#if defined(ENABLE_AVX)
#define MS_SIMD_RUN_AVX(function, ...) MS_EXPAND(function(256, 8, __VA_ARGS__))
#else
#define MS_SIMD_RUN_AVX(function, ...)
#endif

// enable sse
#if defined(ENABLE_SSE)
#define MS_SIMD_RUN_SSE(function, ...) MS_EXPAND(function(128, 4, __VA_ARGS__))
#else
#define MS_SIMD_RUN_SSE(function, ...)
#endif

// enable neon
#if defined(ENABLE_NEON)
#define MS_SIMD_RUN_NEON(function, ...) MS_EXPAND(function(128, 4, __VA_ARGS__))
#else
#define MS_SIMD_RUN_NEON(function, ...)
#endif

// enable neon/sse
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
#define MS_SIMD_RUN_SSEORNEON128(function, ...) MS_EXPAND(function(128, 4, __VA_ARGS__))
#else
#define MS_SIMD_RUN_SSEORNEON128(function, ...)
#endif

// scalar (c style data)
#define MS_SIMD_RUN_SCALAR(function, ...) MS_EXPAND(function(32, 1, __VA_ARGS__))

#define MS_SIMD_RUN(function, ...)                   \
  do {                                               \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__);       \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);          \
    MS_SIMD_RUN_SSEORNEON128(function, __VA_ARGS__); \
    MS_SIMD_RUN_SCALAR(function, __VA_ARGS__);       \
  } while (0)

#define MS_SIMD_RUN_NO_SCALAR(function, ...)         \
  do {                                               \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__);       \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);          \
    MS_SIMD_RUN_SSEORNEON128(function, __VA_ARGS__); \
  } while (0)

#define MS_SIMD_RUN_X86(function, ...)         \
  do {                                         \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__); \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);    \
    MS_SIMD_RUN_SSE(function, __VA_ARGS__);    \
    MS_SIMD_RUN_SCALAR(function, __VA_ARGS__); \
  } while (0)

#define MS_SIMD_RUN_X86_NO_SCALAR(function, ...) \
  do {                                           \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__);   \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);      \
    MS_SIMD_RUN_SSE(function, __VA_ARGS__);      \
  } while (0)

#endif  // MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
