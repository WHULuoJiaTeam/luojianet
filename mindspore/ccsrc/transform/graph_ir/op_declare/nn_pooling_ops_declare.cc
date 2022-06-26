/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/nn_pooling_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// MaxPool
INPUT_MAP(MaxPool) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MaxPool) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                     {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPool) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPool, kNameMaxPool, ADPT_DESC(MaxPool))

// MaxPool3D
INPUT_MAP(MaxPool3D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MaxPool3D) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                       {"pad_list", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"dilation", ATTR_DESC(dilation, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"ceil_mode", ATTR_DESC(ceil_mode, AnyTraits<int64_t>())},
                       {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPool3D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPool3D, kNameMaxPool3D, ADPT_DESC(MaxPool3D))

// MaxPool3DGrad
INPUT_MAP(MaxPool3DGrad) = {{1, INPUT_DESC(orig_x)}, {2, INPUT_DESC(orig_y)}, {3, INPUT_DESC(grads)}};
ATTR_MAP(MaxPool3DGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                           {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                           {"pad_list", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                           {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPool3DGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPool3DGrad, kNameMaxPool3DGrad, ADPT_DESC(MaxPool3DGrad))

// MaxPool3DGradGrad
INPUT_MAP(MaxPool3DGradGrad) = {{1, INPUT_DESC(orig_x)}, {2, INPUT_DESC(orig_y)}, {3, INPUT_DESC(grads)}};
ATTR_MAP(MaxPool3DGradGrad) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPool3DGradGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPool3DGradGrad, kNameMaxPool3DGradGrad, ADPT_DESC(MaxPool3DGradGrad))

// AvgPool
INPUT_MAP(AvgPool) = {{1, INPUT_DESC(x)}};
ATTR_MAP(AvgPool) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                     {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(AvgPool) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AvgPool, kNameAvgPool, ADPT_DESC(AvgPool))

// MaxPoolGrad
INPUT_MAP(MaxPoolGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grad)}};
ATTR_MAP(MaxPoolGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                         {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPoolGrad, kNameMaxPoolGrad, ADPT_DESC(MaxPoolGrad))

// MaxPoolGradGrad
INPUT_MAP(MaxPoolGradGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grad)}};
ATTR_MAP(MaxPoolGradGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                             {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                             {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                             {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolGradGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPoolGradGrad, kNameMaxPoolGradGrad, ADPT_DESC(MaxPoolGradGrad))

// avgpoolgrad
INPUT_MAP(AvgPoolGrad) = {{1, INPUT_DESC(orig_input_shape)}, {2, INPUT_DESC(input_grad)}};
ATTR_MAP(AvgPoolGrad) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                         {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())},
                         {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(AvgPoolGrad) = {{0, OUTPUT_DESC(out_grad)}};
REG_ADPT_DESC(AvgPoolGrad, kNameAvgPoolGrad, ADPT_DESC(AvgPoolGrad))

// MaxPoolWithArgmax
INPUT_MAP(MaxPoolWithArgmax) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MaxPoolWithArgmax) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolWithArgmax) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(argmax)}};
REG_ADPT_DESC(MaxPoolWithArgmax, kNameMaxPoolWithArgmax, ADPT_DESC(MaxPoolWithArgmax))

// MaxPoolGradWithArgmax
INPUT_MAP(MaxPoolGradWithArgmax) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}, {3, INPUT_DESC(argmax)}};
ATTR_MAP(MaxPoolGradWithArgmax) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolGradWithArgmax) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPoolGradWithArgmax, kNameMaxPoolGradWithArgmax, ADPT_DESC(MaxPoolGradWithArgmax))

// MaxPoolGradGradWithArgmax
INPUT_MAP(MaxPoolGradGradWithArgmax) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}, {3, INPUT_DESC(argmax)}};
ATTR_MAP(MaxPoolGradGradWithArgmax) = {
  {"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_mode", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolGradGradWithArgmax) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPoolGradGradWithArgmax, kNameMaxPoolGradGradWithArgmax, ADPT_DESC(MaxPoolGradGradWithArgmax))

// Pooling
INPUT_MAP(Pooling) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Pooling) = {{"mode", ATTR_DESC(mode, AnyTraits<int64_t>())},
                     {"global", ATTR_DESC(global_pooling, AnyTraits<bool>())},
                     {"kernel_size", ATTR_DESC(window, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(stride, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"pad", ATTR_DESC(pad, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"dilation", ATTR_DESC(dilation, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                     {"round_mode", ATTR_DESC(ceil_mode, AnyTraits<int64_t>())},
                     {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(Pooling) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Pooling, kNamePooling, ADPT_DESC(Pooling))

// MaxPoolV3
INPUT_MAP(MaxPoolV3) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MaxPoolV3) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"padding_mode", ATTR_DESC(padding_mode, AnyTraits<std::string>())},
                       {"pad", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                       {"global", ATTR_DESC(global_pooling, AnyTraits<bool>())},
                       {"ceil_mode", ATTR_DESC(ceil_mode, AnyTraits<bool>())}};
OUTPUT_MAP(MaxPoolV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaxPoolV3, kNameMaxPoolV3, ADPT_DESC(MaxPoolV3))

// AvgPoolV2
INPUT_MAP(AvgPoolV2) = {{1, INPUT_DESC(x)}};
ATTR_MAP(AvgPoolV2) = {{"kernel_size", ATTR_DESC(ksize, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"padding_mode", ATTR_DESC(padding_mode, AnyTraits<std::string>())},
                       {"pad", ATTR_DESC(pads, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                       {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                       {"global", ATTR_DESC(global_pooling, AnyTraits<bool>())},
                       {"ceil_mode", ATTR_DESC(ceil_mode, AnyTraits<bool>())}};
OUTPUT_MAP(AvgPoolV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AvgPoolV2, kNameAvgPoolV2, ADPT_DESC(AvgPoolV2))

// GlobalAveragePool
INPUT_MAP(GlobalAveragePool) = {{1, INPUT_DESC(x)}};
ATTR_MAP(GlobalAveragePool) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GlobalAveragePool) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GlobalAveragePool, kNameGlobalAvgPool, ADPT_DESC(GlobalAveragePool))

// Upsample
INPUT_MAP(Upsample) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Upsample) = {{"scale", ATTR_DESC(scale, AnyTraits<float>())},
                      {"stride_h", ATTR_DESC(stride_h, AnyTraits<int64_t>())},
                      {"stride_w", ATTR_DESC(stride_w, AnyTraits<int64_t>())}};
OUTPUT_MAP(Upsample) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Upsample, kNameUpsample, ADPT_DESC(Upsample))
}  // namespace mindspore::transform
