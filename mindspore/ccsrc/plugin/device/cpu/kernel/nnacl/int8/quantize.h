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

#ifndef MINDSPORE_NNACL_QUANTIZATION_QUANTIZE_H_
#define MINDSPORE_NNACL_QUANTIZATION_QUANTIZE_H_

#include <math.h>
#include <limits.h>
#include "nnacl/op_base.h"

#define INPUT_PER_CHANNEL 0b001
#define FILTER_PER_CHANNEL 0b010
#define OUTPUT_PER_CHANNEL 0b100

typedef struct ConvQuantArg {
  RoundingMode round_mode_;
  CalFixedMultiplierMode quant_multiplier_mode_;
  QuantArg *input_quant_args_;
  QuantArg *filter_quant_args_;
  QuantArg *output_quant_args_;
  double *real_multiplier_;
  int32_t *left_shift_;
  int32_t *right_shift_;
  int32_t *quant_multiplier_;
  int32_t *out_act_min_;
  int32_t *out_act_max_;
  size_t input_arg_num_;
  size_t filter_arg_num_;
  size_t output_arg_num_;
  uint8_t per_channel_;
} ConvQuantArg;

typedef struct ConcatQuantArg {
  QuantArg *in_args_;
  QuantArg out_args_;
  int8_t output_activation_min_;
  int8_t output_activation_max_;
} ConcatQuantArg;

typedef struct PreluQuantArg {
  int *input_sizes_;
  int output_size_;
  int **input_shapes_;
  int *output_shape_;
  size_t input_num_;
  size_t output_dim_;
  float alpha_;
  QuantArg in_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
  QuantArg *in_quant_args_;
  QuantArg out_quant_args_;
} PreluQuantArg;

typedef struct CropQuantArg {
  QuantArg in_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
} CropQuantArg;

typedef struct ArithSelfQuantArg {
  QuantArg in_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
  int output_multiplier_;
  int shift_left_;
  int shift_right_;
} ArithSelfQuantArg;

typedef struct GatherQuantArg {
  double alpha_;
  int zp_in_;
  int zp_out_;
} GatherQuantArg;

typedef struct DynamicGatherQuantArg {
  float *scale_in_;
  int *zp_in_;
} DynamicGatherQuantArg;

typedef struct SoftmaxQuantArg {
  QuantArg in_quant_args_;
  QuantArg out_quant_arg_;
  int output_activation_min_;
  int output_activation_max_;
  int output_multiplier_;
  int shift_left_;
  int shift_right_;
} SoftmaxQuantArg;

typedef struct SubQuantArg {
  QuantArg in0_args_;
  QuantArg in1_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
  int input0_multiplier_;
  int input1_multiplier_;
  int output_multiplier_;
  int input0_shift_;
  int input1_shift_;
  int output_shift_;
  int left_shift_result0_;
  int left_shift_result1_;
  int right_shift0_;
  int right_shift1_;
  int left_shift_out_;
  int right_shift_out_;
} SubQuantArg;

typedef struct ArithmeticQuantArg {
  QuantArg in0_args_;
  QuantArg in1_args_;
  QuantArg out_args_;
} ArithmeticQuantArg;

typedef struct DivQuantArg {
  QuantArg in0_args_;
  QuantArg in1_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
  int output_multiplier_;
  int output_shift_;
} DivQuantArg;

typedef struct ReduceQuantArg {
  double in_scale_;
  int32_t in_zp_;
  double out_scale_;
  int32_t out_zp_;
  int32_t in_out_multiplier_;
  int in_out_left_shift_;
  int in_out_right_shift_;
  int32_t mean_multiplier_;
  int mean_left_shift_;
  int mean_right_shift_;
  int32_t prod_multiplier_;
  int prod_left_shift_;
  int prod_right_shift_;
  int32_t sum_square_multiplier_;
  int sum_square_left_shift_;
  int sum_square_right_shift_;
} ReduceQuantArg;

typedef struct LeakyReluQuantArg {
  OpParameter op_parameter_;
  PreluQuantArg quant_arg;
  float slope_;
  int64_t axis_;
  int *in_shape_;
  int *out_shape_;
  int input_dim_;
  int element_num;
} LeakyReluQuantArg;

typedef struct ResizeQuantArg {
  int32_t ratio_x_;
  int32_t ratio_y_;
  int32_t *x_axis_index_;
  int32_t *x_axis_lower_;
  int32_t *x_axis_upper_;
  int32_t *y_axis_index_;
  int32_t *y_axis_lower_;
  int32_t *y_axis_upper_;
} ResizeQuantArg;

typedef struct ResizeFloatScaleQuantArg {
  float ratio_x_;
  float ratio_y_;
  float *x_axis_index_;
  int32_t *x_axis_lower_;
  int32_t *x_axis_upper_;
  float *y_axis_index_;
  int32_t *y_axis_lower_;
  int32_t *y_axis_upper_;
} ResizeFloatScaleQuantArg;

#ifdef __cplusplus
extern "C" {
#endif

void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

void QuantizeMultiplierSmallerThanOne(double double_multiplier, int32_t *quantized_multiplier, int *right_shift);

void QuantizeRoundParameterWithDoublePrecision(double double_multiplier, int32_t *quantized_multiplier, int *left_shift,
                                               int *right_shift);

void QuantizeRoundParameterWithSinglePrecision(double double_multiplier, int32_t *quantized_multiplier, int *left_shift,
                                               int *right_shift);

uint8_t QuantizeToUint8(float real_value, float scale, int32_t zp);

int32_t QuantizeToInt8(float real_value, float scale, int32_t zp);

void CalculateActivationRangeQuantized(bool is_relu, bool is_relu6, int32_t zp, float scale, int *mini, int *maxi);
// quantize from float to int8
void Quantize(const float *input_data, int length, float scale, int zero_point, int8_t *output_data);

// dequantize from int8 to float
void Dequantize(const int8_t *input_data, int length, float scale, int zero_point, float *output_data);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_QUANTIZATION_QUANTIZE_H_
