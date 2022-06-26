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

#include "ops/strided_slice.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <bitset>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace luojianet_ms {
namespace ops {
namespace {
std::vector<int64_t> TenToTwo(int64_t num) {
  std::vector<int64_t> output;
  if (num == 0) {
    output.push_back(0);
    return output;
  }
  while (num) {
    output.push_back(num % 2);
    num /= 2;
  }

  return output;
}

void GetAndCheckAttrMask(const PrimitivePtr &primitive, std::vector<int64_t> *begin_pos, std::vector<int64_t> *end_pos,
                         std::vector<int64_t> *ellipsis_pos, std::vector<int64_t> *new_axis_pos,
                         std::vector<int64_t> *shrink_axis_pos, bool slice_dynamic) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto begin_mask = GetValue<int64_t>(primitive->GetAttr(kBeginMask));
  auto end_mask = GetValue<int64_t>(primitive->GetAttr(kEndMask));
  auto ellipsis_mask = GetValue<int64_t>(primitive->GetAttr(kEllipsisMask));
  auto new_axis_mask = GetValue<int64_t>(primitive->GetAttr(kNewAxisMask));
  auto shrink_axis_mask = GetValue<int64_t>(primitive->GetAttr(kShrinkAxisMask));

  *begin_pos = TenToTwo(begin_mask);
  *end_pos = TenToTwo(end_mask);
  *ellipsis_pos = TenToTwo(ellipsis_mask);
  *new_axis_pos = TenToTwo(new_axis_mask);
  *shrink_axis_pos = TenToTwo(shrink_axis_mask);
  return;
}

int64_t GetSlicingLengthForPositiveStrides(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) {
  int64_t slicing_length = 0;
  if ((start_pos < x_dim) && end_pos >= -x_dim) {
    if (-x_dim <= start_pos && start_pos < 0) {
      start_pos += x_dim;
    }
    if (start_pos < -x_dim) {
      start_pos = 0;
    }
    if (-x_dim <= end_pos && end_pos < 0) {
      end_pos += x_dim;
    }
    if (end_pos > x_dim) {
      end_pos = x_dim;
    }
    if (start_pos > end_pos) {
      slicing_length = 0;
    } else {
      slicing_length = 1 + (end_pos - 1 - start_pos) / strides;
    }
  }
  return slicing_length;
}

int64_t GetSlicingLengthForNegativeStrides(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) {
  int64_t slicing_length = 0;
  if (start_pos >= -x_dim && end_pos < x_dim) {
    if (start_pos > 0 && start_pos < x_dim) {
      start_pos += -x_dim;
    }
    if (start_pos >= x_dim) {
      start_pos = -1;
    }
    if (end_pos >= 0 && end_pos < x_dim) {
      end_pos += -x_dim;
    }
    if (end_pos < -x_dim - 1) {
      end_pos = -x_dim - 1;
    }
    if (start_pos <= end_pos) {
      slicing_length = 0;
    } else {
      slicing_length = 1 + (end_pos + 1 - start_pos) / strides;
    }
  }
  return slicing_length;
}

int64_t ComputeSlicingLength(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) {
  int64_t slicing_length = 0;
  if (strides == 0) {
    MS_EXCEPTION(ValueError) << "StridedSlice's input strides cannot contain 0.";
  }
  if (strides > 0) {
    slicing_length = GetSlicingLengthForPositiveStrides(start_pos, end_pos, strides, x_dim);
  } else {
    slicing_length = GetSlicingLengthForNegativeStrides(start_pos, end_pos, strides, x_dim);
  }
  return slicing_length;
}

void EllipsisInferShape(const PrimitivePtr &primitive, const std::vector<int64_t> &x_shape,
                        const std::vector<int64_t> &begin_v, const std::vector<int64_t> &end_v,
                        const std::vector<int64_t> &strides_v, std::vector<int64_t> *infer_shape, size_t i, size_t j,
                        bool has_ellipsis) {
  if (!has_ellipsis) {
    return;
  }
  MS_EXCEPTION_IF_NULL(primitive);
  size_t x_rank = x_shape.size();
  size_t slice_len = begin_v.size();
  std::vector<int64_t> begin_pos;
  std::vector<int64_t> end_pos;
  std::vector<int64_t> ellipsis_pos;
  std::vector<int64_t> new_axis_pos;
  std::vector<int64_t> shrink_axis_pos;
  GetAndCheckAttrMask(primitive, &begin_pos, &end_pos, &ellipsis_pos, &new_axis_pos, &shrink_axis_pos, false);
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(new_axis_pos.size()), kGreaterEqual,
                                           SizeToLong(slice_len), primitive->name());

  size_t num = 0;
  for (size_t n = j + 1; n < slice_len; n++) {
    if (new_axis_pos[n] == 1) {
      num++;
    }
  }

  size_t ellipsis_occupied_dims = x_rank - i - (slice_len - (j + 1)) + num;
  MS_EXCEPTION_IF_NULL(infer_shape);
  (void)infer_shape->insert(infer_shape->end(), x_shape.begin() + LongToSize(i),
                            x_shape.begin() + SizeToLong(i + ellipsis_occupied_dims));
  j += 1;
  i += ellipsis_occupied_dims;

  while (i < x_rank || j < slice_len) {
    int64_t x_dim_size = x_shape[i];
    int64_t start = begin_v[j];
    int64_t finish = end_v[j];
    int64_t strides = strides_v[j];
    if (j < begin_pos.size() || j < slice_len) {
      start = strides_v[j] < 0 ? -1 : 0;
    }
    if (j < end_pos.size() && end_pos[j] == 1) {
      finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
    }
    if (j < new_axis_pos.size() && new_axis_pos[j] == 1) {
      infer_shape->push_back(1);
      j += 1;
      continue;
    }
    if (j < shrink_axis_pos.size() && shrink_axis_pos[j] == 1) {
      if ((-x_shape[i] <= start && start < x_shape[i]) || strides < 0) {
        MS_EXCEPTION(ValueError)
          << "For '" << primitive->name()
          << "', when shrink axis, the 'strides' should be greater than or equal to one, but got " << strides << ".";
      }
      j += 1;
      i += 1;
      continue;
    }
    int64_t slicing_length = ComputeSlicingLength(start, finish, strides, x_dim_size);
    infer_shape->push_back(slicing_length);
    i += 1;
    j += 1;
  }
  return;
}

std::vector<int64_t> ComputeInferShape(const PrimitivePtr &primitive, const std::vector<int64_t> &begin_v,
                                       const std::vector<int64_t> &end_v, const std::vector<int64_t> &strides_v,
                                       const std::vector<int64_t> &x_shape) {
  std::vector<int64_t> begin_pos;
  std::vector<int64_t> end_pos;
  std::vector<int64_t> ellipsis_pos;
  std::vector<int64_t> new_axis_pos;
  std::vector<int64_t> shrink_axis_pos;
  GetAndCheckAttrMask(primitive, &begin_pos, &end_pos, &ellipsis_pos, &new_axis_pos, &shrink_axis_pos, true);

  size_t i = 0;
  size_t j = 0;
  int64_t start;
  int64_t finish;
  int64_t strides;
  int64_t slicing_length;
  bool has_ellipsis = false;
  std::vector<int64_t> infer_shape;
  size_t slice_len = begin_v.size();
  size_t x_rank = x_shape.size();
  while (i < x_rank || j < slice_len) {
    int64_t x_dim_size = x_shape[i];
    if (j < slice_len) {
      start = begin_v[j];
      finish = end_v[j];
      strides = strides_v[j];
      if (j < ellipsis_pos.size() && ellipsis_pos[j] == 1) {
        has_ellipsis = true;
        break;
      }
      if (j < begin_pos.size() && begin_pos[j] == 1) {
        start = strides_v[j] < 0 ? -1 : 0;
      }
      if (j < end_pos.size() && end_pos[j] == 1) {
        finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
      }
      if (j < new_axis_pos.size() && new_axis_pos[j] == 1) {
        infer_shape.push_back(1);
        j += 1;
        continue;
      }
      if (j < shrink_axis_pos.size() && shrink_axis_pos[j] == 1) {
        if (!(-x_shape[i] <= start && start < x_shape[i]) || strides < 0) {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                   << "', when shrink axis, the 'strides'  greater than or equal to one, but got "
                                   << strides << ".";
        }
        j += 1;
        i += 1;
        continue;
      }
    } else {
      start = 0;
      finish = x_shape[i];
      strides = 1;
    }
    slicing_length = ComputeSlicingLength(start, finish, strides, x_dim_size);
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }
  EllipsisInferShape(primitive, x_shape, begin_v, end_v, strides_v, &infer_shape, i, j, has_ellipsis);
  return infer_shape;
}

std::vector<int64_t> DynamicComputeInferShape(const PrimitivePtr &primitive, const std::vector<int64_t> &begin_v,
                                              const std::vector<int64_t> &end_v, const std::vector<int64_t> &strides_v,
                                              const std::vector<int64_t> &x_shape, const size_t slice_len) {
  // currently not support mask
  std::vector<int64_t> begin_pos;
  std::vector<int64_t> end_pos;
  std::vector<int64_t> ellipsis_pos;
  std::vector<int64_t> new_axis_pos;
  std::vector<int64_t> shrink_axis_pos;
  GetAndCheckAttrMask(primitive, &begin_pos, &end_pos, &ellipsis_pos, &new_axis_pos, &shrink_axis_pos, false);

  size_t i = 0;
  size_t j = 0;
  int64_t start;
  int64_t finish;
  int64_t strides;
  std::vector<int64_t> infer_shape;
  size_t x_rank = x_shape.size();
  while (i < x_rank || j < slice_len) {
    int64_t slicing_length = -1;
    int64_t x_dim_size = x_shape[i];
    if (x_dim_size == 1) {
      slicing_length = 1;
    }
    if (j >= slice_len && x_dim_size > 0) {
      start = 0;
      finish = x_shape[i];
      strides = 1;
      if (finish > 0) {
        slicing_length = ComputeSlicingLength(start, finish, strides, x_dim_size);
      }
    }
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }
  return infer_shape;
}

bool CheckAndGetDynamicSlice(const AbstractBasePtr &input_arg, const std::string &arg_name, ShapeVector *slice_value,
                             size_t *slice_len) {
  bool is_dynamic = false;
  MS_EXCEPTION_IF_NULL(input_arg);
  auto input_value = input_arg->BuildValue();
  MS_EXCEPTION_IF_NULL(input_value);
  if (input_arg->isa<abstract::AbstractTuple>()) {
    *slice_value = CheckAndConvertUtils::CheckTupleInt(arg_name, input_value, "StridedSlice");
    *slice_len = (*slice_value).size();
  } else if (input_arg->isa<abstract::AbstractTensor>()) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid(arg_name, input_arg->BuildType(), {kInt64}, "StridedSlice");
    if (input_value->isa<tensor::Tensor>()) {
      *slice_value = CheckAndConvertUtils::CheckTensorIntValue(arg_name, input_value, "StridedSlice");
      *slice_len = (*slice_value).size();
    } else {
      // slice is AnyValue
      is_dynamic = true;
      auto slice_shape = CheckAndConvertUtils::GetTensorInputShape("StridedSlice", {input_arg}, 0);
      if (slice_shape->shape().size() != 1) {
        MS_EXCEPTION(ValueError) << "For 'StridedSlice', " << arg_name << " must be 1-D, but got"
                                 << slice_shape->shape().size() << "-D.";
      }
      *slice_len = LongToSize(slice_shape->shape()[0]);
    }
  } else {
    MS_EXCEPTION(TypeError) << "For StridedSlice, " << arg_name << " must be tuple or Tensor.";
  }

  if (arg_name == "strides") {
    if (std::any_of((*slice_value).begin(), (*slice_value).end(),
                    [](int64_t stride_value) { return stride_value == 0; })) {
      MS_EXCEPTION(ValueError) << "StridedSlice's input strides cannot contain 0.";
    }
  }
  return is_dynamic;
}

abstract::ShapePtr StridedSliceInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const size_t x_index = 0;
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[x_index]->BuildShape());
  auto x_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  bool x_is_dyn =
    std::any_of(x_shape.begin(), x_shape.end(), [](int64_t value) { return value == abstract::Shape::SHP_ANY; });
  if (x_is_dyn && (min_shape.size() == 0 || max_shape.size() == 0)) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << ", input x dynamic shape is currently not supported if min and max shapes are not given.";
  }
  ShapeVector begin_v;
  ShapeVector end_v;
  ShapeVector strides_v;
  ShapeVector ret_in_shape;
  size_t begin_len = 0;
  size_t end_len = 0;
  size_t stride_len = 0;
  const size_t begin_index = 1;
  const size_t end_index = 2;
  const size_t stride_index = 3;
  bool begin_dynamic = CheckAndGetDynamicSlice(input_args[begin_index], "begin", &begin_v, &begin_len);
  bool end_dynamic = CheckAndGetDynamicSlice(input_args[end_index], "end", &end_v, &end_len);
  bool stride_dynamic = CheckAndGetDynamicSlice(input_args[stride_index], "strides", &strides_v, &stride_len);
  if (begin_len != stride_len || end_len != stride_len) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", 'begin', 'end' and 'strides' must be the same length, "
                             << "but got length of 'begin': " << begin_len << ", 'end': " << end_len
                             << ", 'strides': " << stride_len;
  }
  bool slice_dynamic = false;
  if (begin_dynamic || end_dynamic || stride_dynamic || x_is_dyn) {
    slice_dynamic = true;
  }
  if (!slice_dynamic) {
    ret_in_shape = ComputeInferShape(primitive, begin_v, end_v, strides_v, x_shape);
    bool has_zero_shape = std::any_of(ret_in_shape.begin(), ret_in_shape.end(), [](int64_t i) { return i == 0; });
    if (has_zero_shape) {
      MS_LOG(EXCEPTION) << "StridedSlice haven't support zero shape yet, now the out shape is " << ret_in_shape;
    }
    return std::make_shared<abstract::Shape>(ret_in_shape);
  }
  ret_in_shape = DynamicComputeInferShape(primitive, begin_v, end_v, strides_v, x_shape, begin_len);
  ShapeVector ret_min_shape(x_shape.size(), 1);
  ShapeVector ret_max_shape = x_shape;
  for (size_t i = 0; i < ret_in_shape.size(); i++) {
    if (ret_in_shape[i] > 0) {
      ret_min_shape[i] = ret_in_shape[i];
      ret_max_shape[i] = ret_in_shape[i];
    } else {
      ret_min_shape[i] = min_shape[i];
      ret_max_shape[i] = max_shape[i];
    }
  }
  return std::make_shared<abstract::Shape>(ret_in_shape, ret_min_shape, ret_max_shape);
}

TypePtr StridedSliceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const size_t x_index = 0;
  return CheckAndConvertUtils::GetTensorInputType(primitive->name(), input_args, x_index);
}
}  // namespace

MIND_API_BASE_IMPL(StridedSlice, PrimitiveC, BaseOperator);
void StridedSlice::set_begin_mask(int64_t begin_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kBeginMask, begin_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kBeginMask, api::MakeValue(begin_mask));
}
int64_t StridedSlice::get_begin_mask() const {
  auto value_ptr = GetAttr(kBeginMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_end_mask(int64_t end_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEndMask, end_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kEndMask, api::MakeValue(end_mask));
}
int64_t StridedSlice::get_end_mask() const {
  auto value_ptr = GetAttr(kEndMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_ellipsis_mask(int64_t ellipsis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEllipsisMask, ellipsis_mask, kGreaterEqual, 0, this->name());
  std::bitset<sizeof(int64_t) * 8> bs(ellipsis_mask);
  std::ostringstream buffer;
  if (bs.count() > 1) {
    buffer << "For" << this->name() << ", only support one ellipsis in the index, but got " << this->get_end_mask();
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  (void)this->AddAttr(kEllipsisMask, api::MakeValue(ellipsis_mask));
}
int64_t StridedSlice::get_ellipsis_mask() const {
  auto value_ptr = GetAttr(kEllipsisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_new_axis_mask(int64_t new_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kNewAxisMask, new_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kNewAxisMask, api::MakeValue(new_axis_mask));
}
int64_t StridedSlice::get_new_axis_mask() const {
  auto value_ptr = GetAttr(kNewAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_shrink_axis_mask(int64_t shrink_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kShrinkAxisMask, shrink_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kShrinkAxisMask, api::MakeValue(shrink_axis_mask));
}
int64_t StridedSlice::get_shrink_axis_mask() const {
  auto value_ptr = GetAttr(kShrinkAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::Init(int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask,
                        int64_t shrink_axis_mask) {
  this->set_begin_mask(begin_mask);
  this->set_end_mask(end_mask);
  this->set_ellipsis_mask(ellipsis_mask);
  this->set_new_axis_mask(new_axis_mask);
  this->set_shrink_axis_mask(shrink_axis_mask);
}

AbstractBasePtr StridedSliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  return std::make_shared<abstract::AbstractTensor>(StridedSliceInferType(primitive, input_args),
                                                    StridedSliceInferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameStridedSlice, StridedSlice);
}  // namespace ops
}  // namespace luojianet_ms
