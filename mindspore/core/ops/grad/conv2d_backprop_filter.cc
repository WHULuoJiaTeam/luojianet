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

#include <set>
#include <map>
#include <memory>

#include "ops/grad/conv2d_backprop_filter.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv2DBackpropFilterDoutIndex = 0;
constexpr size_t kConv2DBackpropFilterInputIndex = 1;

abstract::ShapePtr Conv2DBackpropFilterInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  std::vector<int64_t> out_shape;
  abstract::ShapePtr ret_shape;
  constexpr size_t kFilterSizeIndex = 2;
  auto filter_size = input_args[kFilterSizeIndex];
  auto filter_size_v = filter_size->BuildValue();
  MS_EXCEPTION_IF_NULL(filter_size_v);

  if (filter_size->isa<abstract::AbstractTensor>()) {
    if (filter_size_v->isa<tensor::Tensor>()) {
      out_shape = CheckAndConvertUtils::CheckTensorIntValue("filter size", filter_size_v, prim_name);
      ret_shape = std::make_shared<abstract::Shape>(out_shape);
    } else {
      auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kFilterSizeIndex);
      MS_EXCEPTION_IF_NULL(shape_ptr);
      auto shape_shape = shape_ptr->shape();
      if (shape_shape.size() != 1) {
        MS_LOG(EXCEPTION) << "For '" << prim_name << "', filter size must be 1-D, but got " << shape_shape.size();
      }

      auto abstract_tensor = filter_size->cast<abstract::AbstractTensorPtr>();
      MS_EXCEPTION_IF_NULL(abstract_tensor);
      auto shape_max_value = abstract_tensor->get_max_value();
      auto shape_min_value = abstract_tensor->get_min_value();
      if (shape_max_value == nullptr || shape_min_value == nullptr) {
        MS_LOG(EXCEPTION) << "For '" << prim_name
                          << "', Max_value or min value of 'filter' size can not be empty when its value is dynamic.";
      }

      auto shape_max = GetValue<std::vector<int64_t>>(shape_max_value);
      auto shape_min = GetValue<std::vector<int64_t>>(shape_min_value);

      auto filter_len = LongToSize(shape_shape[0]);
      if (shape_max.size() != filter_len || shape_min.size() != filter_len) {
        MS_LOG(EXCEPTION) << "For " << prim_name << ", filter size's min or max value is valid.";
      }

      for (size_t i = 0; i < filter_len; i++) {
        if (shape_min[i] == shape_max[i]) {
          out_shape.push_back(shape_min[i]);
        } else {
          out_shape.push_back(abstract::Shape::SHP_ANY);
        }
      }
      ret_shape = std::make_shared<abstract::Shape>(out_shape, shape_min, shape_max);
    }
  } else if (filter_size->isa<abstract::AbstractTuple>()) {
    // check tensor, tuple or int to raise error.
    out_shape = CheckAndConvertUtils::CheckTupleInt("input[filter_size]", filter_size_v, prim_name);
    ret_shape = std::make_shared<abstract::Shape>(out_shape);
  } else {
    auto size_type = filter_size->BuildType();
    MS_EXCEPTION_IF_NULL(size_type);
    MS_EXCEPTION(TypeError) << "The primitive[" << prim_name << "]'s input[filter size] must be a tuple or Tensor, "
                            << "but got " << size_type->ToString();
  }
  return ret_shape;
}

TypePtr Conv2DBackpropFilterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // check
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kConv2DBackpropFilterInputIndex]->BuildType());
  (void)types.emplace("doutput", input_args[kConv2DBackpropFilterDoutIndex]->BuildType());
  std::set<TypePtr> valid_x_type = {kInt8, kInt32, kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_x_type, prim_name);
}
}  // namespace

MIND_API_BASE_IMPL(Conv2DBackpropFilter, PrimitiveC, BaseOperator);
void Conv2DBackpropFilter::Init(const int64_t out_channel, const std::vector<int64_t> &kernel_size,
                                const PadMode &pad_mode, const std::vector<int64_t> &pad_list, const int64_t mode,
                                const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                const int64_t group, const Format &format) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_mode(mode);
  constexpr size_t kStride4dSize = 4;
  if (stride.size() == kStride4dSize) {
    set_stride({stride[2], stride[3]});
  } else {
    set_stride(stride);
  }
  set_dilation(dilation);
  set_group(group);
  set_format(format);
}

void Conv2DBackpropFilter::set_out_channel(const int64_t out_channel) {
  (void)this->AddAttr(kOutChannel, api::MakeValue(out_channel));
}

int64_t Conv2DBackpropFilter::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

std::vector<int64_t> Conv2DBackpropFilter::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode Conv2DBackpropFilter::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void Conv2DBackpropFilter::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> Conv2DBackpropFilter::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_mode(const int64_t mode) { (void)this->AddAttr(kMode, api::MakeValue(mode)); }

int64_t Conv2DBackpropFilter::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_stride(const std::vector<int64_t> &stride) {
  (void)this->AddAttr(kStride, api::MakeValue(stride));
}

std::vector<int64_t> Conv2DBackpropFilter::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_dilation(const std::vector<int64_t> &dilation) {
  (void)this->AddAttr(kDilation, api::MakeValue(dilation));
}

std::vector<int64_t> Conv2DBackpropFilter::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_group(const int64_t group) { (void)this->AddAttr(kGroup, api::MakeValue(group)); }

int64_t Conv2DBackpropFilter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

Format Conv2DBackpropFilter::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr Conv2DBackpropFilterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto res = std::make_shared<abstract::AbstractTensor>(Conv2DBackpropFilterInferType(primitive, input_args),
                                                        Conv2DBackpropFilterInferShape(primitive, input_args));
  return res;
}
REGISTER_PRIMITIVE_EVAL_IMPL(Conv2DBackpropFilter, prim::kPrimConv2DBackpropFilter, Conv2DBackpropFilterInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
