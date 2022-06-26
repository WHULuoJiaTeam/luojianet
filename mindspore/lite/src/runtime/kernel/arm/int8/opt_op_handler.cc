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

#include "src/runtime/kernel/arm/int8/opt_op_handler.h"
#include <cstdlib>
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ENABLE_ARM64

void MatMulR4Int8_optimize_handler(const int8_t *a, const int8_t *b, int *dst, int row4, int col4, int deep16,
                                   const int *input_sum, const int *bias) {
  return MatMulOptR4Int8Neon64(a, b, dst, row4, col4, deep16, input_sum, bias);
}

void MatMulRInt8_optimize_handler(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                  size_t stride, const int32_t *input_sum, const int32_t *bias,
                                  const int32_t *left_shift, const int32_t *right_shift, const int32_t *multiplier,
                                  int32_t output_zp, int32_t mini, int32_t maxi, size_t per_channel) {
  return MatmulInt8DpNeon64(a, b, dst, UP_ROUND(row, C8NUM), UP_ROUND(col, C8NUM), deep_4, input_sum, bias, mini, maxi,
                            output_zp, multiplier, left_shift, right_shift, row, col, stride, per_channel);
}
void MatMulDpInt8_optimize_handler(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                   size_t stride, const int32_t *input_sum, const int32_t *bias,
                                   const int32_t *left_shift, const int32_t *right_shift, const int32_t *multiplier,
                                   int32_t output_zp, int32_t mini, int32_t maxi, size_t per_channel,
                                   const int32_t *filter_zp) {
  return MatmulInt8DpOpt(a, b, dst, row, col, deep_4, input_sum, bias, mini, maxi, output_zp, multiplier, left_shift,
                         right_shift, stride, per_channel, filter_zp);
}
#endif

#ifdef __cplusplus
}
#endif
