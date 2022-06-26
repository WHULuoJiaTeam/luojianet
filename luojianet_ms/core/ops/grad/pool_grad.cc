/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/pool_grad.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
MIND_API_BASE_IMPL(PoolGrad, PrimitiveC, BaseOperator);
std::vector<int64_t> PoolGrad::_grad_check_vector(const std::string &arg_name, std::vector<int64_t> arg_val,
                                                  const std::string &op_name) {
  std::vector<int64_t> ret;
  std::string error_msg = "For '" + op_name + "'" + " the '" + arg_name +
                          "' should be a vector of one, two or four "
                          "positive int number(s), but got error arg_val";
  switch ((int64_t)arg_val.size()) {
    case 1:
      ret = {1, 1, arg_val[0], arg_val[0]};
      break;
    case 2:
      ret = {1, 1, arg_val[0], arg_val[1]};
      break;
    case 4:
      ret = arg_val;
      break;
    default:
      MS_LOG(EXCEPTION) << error_msg;
  }
  for (auto it : arg_val) {
    if (it <= 0) {
      MS_LOG(EXCEPTION) << error_msg;
    }
  }
  return ret;
}

void PoolGrad::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                    const PadMode &pad_mode, const Format &format) {
  this->set_kernel_size(kernel_size);
  this->set_strides(strides);
  this->set_pad_mode(pad_mode);
  this->set_format(format);
}

void PoolGrad::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  std::vector<int64_t> k_size = _grad_check_vector(kKernelSize, kernel_size, this->name());
  (void)this->AddAttr(kKernelSize, api::MakeValue(k_size));
}

void PoolGrad::set_strides(const std::vector<int64_t> &strides) {
  std::vector<int64_t> strides_ = _grad_check_vector(kStrides, strides, this->name());
  (void)this->AddAttr(kStrides, api::MakeValue(strides_));
}

void PoolGrad::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

void PoolGrad::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

std::vector<int64_t> PoolGrad::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> PoolGrad::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode PoolGrad::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

Format PoolGrad::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

REGISTER_PRIMITIVE_C(kNamePoolGrad, PoolGrad);
}  // namespace ops
}  // namespace luojianet_ms
