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

#include <memory>

#include "ops/fusion/conv2d_backprop_filter_fusion.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(Conv2DBackpropFilterFusion, PrimitiveC, Conv2DBackpropFilter);
void Conv2DBackpropFilterFusion::Init(const int64_t out_channel, const std::vector<int64_t> &kernel_size,
                                      const PadMode &pad_mode, const std::vector<int64_t> &pad_list, const int64_t mode,
                                      const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                      const int64_t group, const Format &format, const ActivationType activation_type) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_mode(mode);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
  set_activation_type(activation_type);
}

void Conv2DBackpropFilterFusion::set_activation_type(const ActivationType activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}

void Conv2DBackpropFilterFusion::set_in_channel(const int64_t in_channel) {
  (void)this->AddAttr(kInChannel, api::MakeValue(in_channel));
}

ActivationType Conv2DBackpropFilterFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

int64_t Conv2DBackpropFilterFusion::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameConv2DBackpropFilterFusion, Conv2DBackpropFilterFusion);
}  // namespace ops
}  // namespace mindspore
