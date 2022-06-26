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
#include "common/graph_kernel/model/op_node.h"

#include <math.h>
#include <algorithm>
#include <set>
#include <sstream>
#include <vector>
#include <functional>
#include <numeric>

#include "utils/hash_map.h"
#include "common/graph_kernel/model/node.h"

namespace luojianet_ms::graphkernel::inner {
namespace {
constexpr size_t kFirstDataIndex = 0;
constexpr size_t kSecondDataIndex = 1;
constexpr size_t kThirdDataIndex = 2;
}  // namespace

std::vector<int64_t> GetListInt(const ValuePtr &attr_value) {
  bool is_int64 = true;
  auto get_int_value = [&is_int64](const ValuePtr &value) -> int64_t {
    if (value->isa<Int64Imm>()) {
      return GetValue<int64_t>(value);
    }
    is_int64 = false;
    return static_cast<int64_t>(GetValue<int>(value));
  };
  std::vector<int64_t> list_int;
  const auto &vals = attr_value->cast<ValueSequencePtr>()->value();
  (void)std::transform(vals.begin(), vals.end(), std::back_inserter(list_int), get_int_value);
  if (!is_int64) {
    MS_LOG(WARNING) << "Vector type should be 'int64_t' but got 'int'";
  }
  return list_int;
}

void PrimOp::Check(const NodePtrList &inputs, const DAttrs &attrs) {
  CheckShape(inputs, attrs);
  CheckType(inputs, attrs);
  CheckFormat(inputs, attrs);
}

// check all type to be identical
void PrimOp::CheckType(const NodePtrList &inputs, const DAttrs &) {
  TypeId tid = inputs[0]->type;
  for (size_t i = 1; i < inputs.size(); i++) {
    if (inputs[i]->type != tid) {
      MS_LOG(EXCEPTION) << "Incompatible dtype between input " << 0 << " and input " << i;
    }
  }
}

// check all formats are compatible, only DefaultFormat is compatible with others
void PrimOp::CheckFormat(const NodePtrList &inputs, const DAttrs &) {
  DFormat res = inputs[0]->format;
  size_t i = 0;
  for (size_t j = 1; j < inputs.size(); j++) {
    if (inputs[j]->format != res) {
      if (inputs[j]->format != kOpFormat_DEFAULT && res != kOpFormat_DEFAULT) {
        MS_LOG(EXCEPTION) << "Incompatible format between input " << i << " and input " << (j + 1);
      }
      if (res == kOpFormat_DEFAULT && j < (inputs.size() - 1)) {
        res = inputs[j]->format;
        i = j + 1;
      }
    }
  }
}

NodeBase PrimOp::Infer(const NodePtrList &inputs, const DAttrs &attrs) {
  Check(inputs, attrs);
  NodeBase nodebase;
  nodebase.shape = InferShape(inputs, attrs);
  nodebase.type = InferType(inputs, attrs);
  nodebase.format = InferFormat(inputs, attrs);
  return nodebase;
}

std::string PrimOp::ToString() const {
  std::ostringstream oss;
  oss << Node::ToString();
  oss << " = " << this->op_ << "(";
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (inputs_[i]->NodeType() == NType::Primitive) {
      oss << inputs_[i]->Node::ToString();
    } else {
      oss << inputs_[i]->ToString();
    }
    if (i != inputs_.size() - 1) oss << ", ";
  }
  oss << ")";
  std::ostringstream attr_oss;
  bool has_attr = false;
  std::set<std::string> black_list = {"IsFeatureMapInputList", "IsFeatureMapOutput", "output_names", "input_names"};
  for (auto attr : attrs_) {
    if (attr.second != nullptr && black_list.count(attr.first) == 0) {
      if (has_attr) {
        attr_oss << ", ";
      } else {
        has_attr = true;
      }
      attr_oss << attr.first << ": " << attr.second->ToString();
    }
  }
  if (has_attr) {
    oss << "  // attr {" << attr_oss.str() << "}";
  }
  return oss.str();
}

template <typename TM, typename TD>
tensor::TensorPtr CalcByOperator(const NodePtrList &inputs, const std::string &op, TypeId tid) {
  std::vector<TM> inputs_tm;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_tm), [](const NodePtr &i) {
    return *static_cast<TM *>(std::static_pointer_cast<inner::ConstTensorNode>(i)->data()->data_c());
  });

  luojianet_ms::HashMap<std::string, std::function<TM(const std::vector<TM> &)>> func_map = {
    {"Add", [](const std::vector<TM> &n) { return n[0] + n[1]; }},
    {"Sub", [](const std::vector<TM> &n) { return n[0] - n[1]; }},
    {"Mul", [](const std::vector<TM> &n) { return n[0] * n[1]; }},
    {"RealDiv", [](const std::vector<TM> &n) { return n[0] / n[1]; }},
    {"Neg", [](const std::vector<TM> &n) { return TM(0) - n[0]; }},
    {"Reciprocal", [](const std::vector<TM> &n) { return TM(1) / n[0]; }},
    {"Log", [](const std::vector<TM> &n) { return log(n[0]); }},
    {"Exp", [](const std::vector<TM> &n) { return exp(n[0]); }},
    {"Abs", [](const std::vector<TM> &n) { return n[0] <= TM(0) ? (TM(0) - n[0]) : n[0]; }},
    {"Sqrt", [](const std::vector<TM> &n) { return sqrt(n[0]); }},
    {"Rsqrt", [](const std::vector<TM> &n) { return TM(1) / sqrt(n[0]); }},
  };
  if (func_map.find(op) == func_map.end()) {
    return nullptr;
  }
  return std::make_shared<tensor::Tensor>(static_cast<TD>(func_map[op](inputs_tm)), TypeIdToType(tid));
}

NodePtr PrimOp::InferValue(const NodePtrList &inputs, const DAttrs &, const std::string &op) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) return nullptr;
  }
  TypeId output_type = this->type;
  tensor::TensorPtr res = nullptr;
  switch (static_cast<int>(output_type)) {
    case TypeId::kNumberTypeUInt8: {
      res = CalcByOperator<uint8_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CalcByOperator<int8_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CalcByOperator<int16_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CalcByOperator<int32_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      res = CalcByOperator<int64_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt16: {
      res = CalcByOperator<uint16_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt32: {
      res = CalcByOperator<uint32_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt64: {
      res = CalcByOperator<uint64_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      res = CalcByOperator<float16, double>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      res = CalcByOperator<float, double>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat64: {
      res = CalcByOperator<double, double>(inputs, op, output_type);
      break;
    }
    default:
      return nullptr;
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
}

// default format shape to fractal_Nz format shape
DShape ToNz(const DShape &default_shape) {
  constexpr size_t nz_size = 2;
  constexpr auto align16 = 16;
  auto len = default_shape.size();
  DShape leading_shape;
  DShape tail_shape;
  if (default_shape.size() == 1 && default_shape[0] == 1) {
    // # As shape (1,) can broadcast to any shape, it can be regarded as a special FractalNZ shape
    return default_shape;
  }
  if (default_shape.size() > nz_size) {
    (void)leading_shape.insert(leading_shape.end(), default_shape.begin(), default_shape.end() - SizeToLong(nz_size));
  }
  if (default_shape.size() == 1 || (default_shape.size() >= nz_size && default_shape[len - nz_size] == 1)) {
    // (32) or (N, 1, 32) -> (N, 2, 1, 1, 16)
    if (default_shape.back() % align16 != 0) {
      MS_LOG(EXCEPTION) << "default_shape[-1] should be multiplies of 16, but got " << default_shape.back();
    }
    tail_shape = {default_shape.back() / align16, 1, 1, align16};
  } else if (default_shape.size() >= nz_size || default_shape[1] == 1) {
    // (N, 32, 1) -> (N, 1, 2, 16, 1)
    if (default_shape[len - nz_size] % align16 != 0) {
      MS_LOG(EXCEPTION) << "default_shape[-2] should be multiplies of 16, but got " << default_shape[len - nz_size];
    }
    tail_shape = {1, default_shape[0] / align16, align16, 1};
  } else {
    // (N, 32, 48) -> (N, 3, 2, 16, 16)
    if (default_shape.back() % align16 != 0 || default_shape[len - nz_size] % align16 != 0) {
      MS_LOG(EXCEPTION) << "default_shape[-1] and default_shape[-2]should be multiplies of 16, but got "
                        << default_shape.back() << " " << default_shape[len - nz_size];
    }
    tail_shape = {default_shape[1] / align16, default_shape[0] / align16, align16, align16};
  }
  (void)leading_shape.insert(leading_shape.end(), tail_shape.begin(), tail_shape.end());
  return leading_shape;
}

DShape BroadcastShape(const NodePtrList &inputs, bool to_nz = false) {
  std::vector<std::vector<int64_t>> shapes;
  for (auto &input : inputs) {
    if (to_nz && input->format != kOpFormat_FRAC_NZ) {
      (void)shapes.emplace_back(ToNz(input->shape));
    } else {
      (void)shapes.emplace_back(input->shape);
    }
  }
  auto max_dim_input =
    std::max_element(shapes.begin(), shapes.end(),
                     [](const std::vector<int64_t> &a, const std::vector<int64_t> &b) { return a.size() < b.size(); });
  auto max_dim = max_dim_input->size();
  std::vector<std::vector<int64_t>> align_shapes;
  for (auto &s : shapes) {
    std::vector<int64_t> cur(max_dim - s.size(), 1);
    (void)cur.insert(cur.end(), s.begin(), s.end());
    (void)align_shapes.emplace_back(cur);
  }
  std::vector<int64_t> output_shape(max_dim, 1);
  for (size_t i = 0; i < max_dim; i++) {
    for (auto &align_shape : align_shapes) {
      if (align_shape[i] > 1) {
        if (output_shape[i] == 1) {
          output_shape[i] = align_shape[i];
        }
        if (output_shape[i] != align_shape[i]) {
          MS_LOG(EXCEPTION) << "Shape broadcast failed. " << output_shape[i] << " vs " << align_shape[i];
        }
      }
    }
  }
  return output_shape;
}

DShape ElemwiseOp::InferShape(const NodePtrList &inputs, const DAttrs &) {
  if (std::all_of(inputs.begin(), inputs.end(), [](const NodePtr &input) {
        return input->format == kOpFormat_DEFAULT || input->format == kOpFormat_NHWC || input->format == kOpFormat_NCHW;
      })) {
    return BroadcastShape(inputs, false);
  }
  if (std::all_of(inputs.begin(), inputs.end(), [](const NodePtr &input) {
        return input->format == kOpFormat_DEFAULT || input->format == kOpFormat_NHWC ||
               input->format == kOpFormat_NCHW || input->format == kOpFormat_FRAC_NZ;
      })) {
    return BroadcastShape(inputs, true);
  }
  std::string inputs_format;
  for (const auto &input : inputs) {
    static_cast<void>(inputs_format.append(" ").append(input->format));
  }
  MS_LOG(EXCEPTION) << "Unsupported inputs format: " << inputs_format;
}

DFormat ElemwiseOp::InferFormat(const NodePtrList &inputs, const DAttrs &) {
  auto it = std::find_if(inputs.begin(), inputs.end(), [](const NodePtr &i) { return i->format != kOpFormat_DEFAULT; });
  return it == inputs.end() ? kOpFormat_DEFAULT : (*it)->format;
}

TypeId CastOp::InferType(const NodePtrList &, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "dst_type");
  auto dst_type = attrs.find("dst_type")->second;
  if (dst_type->isa<Type>()) {
    return dst_type->cast<TypePtr>()->type_id();
  }
  return StringToTypeId(GetValue<std::string>(dst_type));
}

void SelectOp::CheckType(const NodePtrList &inputs, const DAttrs &) {
  if (inputs[kFirstDataIndex]->type != TypeId::kNumberTypeBool) {
    MS_LOG(EXCEPTION) << "Select's input[0] should be bool type, but got "
                      << TypeIdToString(inputs[kFirstDataIndex]->type, true);
  }
  if (inputs[kSecondDataIndex]->type != inputs[kThirdDataIndex]->type) {
    MS_LOG(EXCEPTION) << "Select's input[1] and input[2]'s type doesn't match: "
                      << TypeIdToString(inputs[kSecondDataIndex]->type, true) << " vs "
                      << TypeIdToString(inputs[kThirdDataIndex]->type, true);
  }
}

DShape ReshapeOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "shape");
  auto new_shape = GetListInt(attrs.find("shape")->second);
  auto origin_shape = inputs[0]->shape;
  auto origin_product = std::accumulate(origin_shape.begin(), origin_shape.end(), 1, std::multiplies<int64_t>());
  auto new_product = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int64_t>());
  if (new_product == 0) {
    new_product = 1;
  }
  for (size_t i = 0; i < new_shape.size(); i++) {
    if (new_shape[i] == -1) {
      new_shape[i] = (origin_product / new_product) * (-1);
      return new_shape;
    }
  }
  if (origin_product != new_product) {
    MS_LOG(EXCEPTION) << "The shape product before and after reshaping should be equal";
  }
  return new_shape;
}

DShape BroadcastToOp::InferShape(const NodePtrList &, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "shape");
  return GetListInt(attrs.find("shape")->second);
}

// check reduce axis in range [-size,size)
void ReduceOp::Check(const NodePtrList &inputs, const DAttrs &attrs) {
  PrimOp::Check(inputs, attrs);
  CHECK_ATTR(attrs, "axis");
  auto axis = GetListInt(attrs.find("axis")->second);
  int64_t size = static_cast<int64_t>(inputs[0]->shape.size());
  auto it = std::find_if(axis.begin(), axis.end(), [&size](const int64_t &i) { return (i >= size || i < (-size)); });
  if (it != axis.end()) {
    MS_LOG(EXCEPTION) << "Reduce axis should be in range [" << (-size) << "," << size << ")"
                      << ", but got " << (*it);
  }
}

DShape ReduceOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "axis");
  CHECK_ATTR(attrs, "keep_dims");
  auto axis = GetListInt(attrs.find("axis")->second);
  auto keepdims = GetValue<bool>(attrs.find("keep_dims")->second);
  if (keepdims) {
    DShape new_shape = inputs[0]->shape;
    for (auto x : axis) {
      new_shape[LongToSize(x)] = 1;
    }
    return new_shape;
  }
  DShape new_shape;
  const auto &input_shape = inputs[0]->shape;
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
      (void)new_shape.emplace_back(input_shape[i]);
    }
  }
  if (new_shape.empty()) {
    (void)new_shape.emplace_back(1);
  }
  return new_shape;
}

DShape Conv2dOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  auto check_nd = [](const std::vector<int64_t> &shape, size_t n) {
    if (shape.size() != n) {
      MS_LOG(EXCEPTION) << "input dimension should be " << n << ", but got  " << shape.size();
    }
  };
  auto shape0 = inputs[0]->shape;
  auto shape1 = inputs[1]->shape;
  constexpr auto dim_len = 4;
  check_nd(shape0, dim_len);
  check_nd(shape1, dim_len);
  CHECK_ATTR(attrs, "format");
  if (inputs[0]->format != kOpFormat_NHWC && inputs[1]->format != kOpFormat_NHWC &&
      GetValue<std::string>(attrs.find("format")->second) != kOpFormat_NHWC) {
    MS_LOG(EXCEPTION) << "check NHWC format failed";
  }
  const auto axis_n = 0;
  const auto axis_h = 1;
  const auto axis_w = 2;
  auto n = shape0[axis_n];
  auto h = shape0[axis_h];
  auto w = shape0[axis_w];
  auto out_channel = shape1[axis_n];
  CHECK_ATTR(attrs, "pad_list");
  CHECK_ATTR(attrs, "pad_mode");
  CHECK_ATTR(attrs, "kernel_size");
  CHECK_ATTR(attrs, "stride");
  CHECK_ATTR(attrs, "dilation");
  auto pad_list = GetListInt(attrs.find("pad_list")->second);
  auto pad_mode = GetValue<std::string>(attrs.find("pad_mode")->second);
  auto kernel_size = GetListInt(attrs.find("kernel_size")->second);
  auto stride = GetListInt(attrs.find("stride")->second);
  auto dilation = GetListInt(attrs.find("dilation")->second);
  check_nd(pad_list, dim_len);
  constexpr auto kernel_len = 2;
  check_nd(kernel_size, kernel_len);
  check_nd(stride, dim_len);
  check_nd(dilation, dim_len);
  bool has_pad = false;
  constexpr auto index1 = 1;
  constexpr auto index2 = 2;
  constexpr auto index3 = 3;
  if (pad_list[0] != pad_list[index1] || pad_list[index2] != pad_list[index3]) {
    has_pad = true;
  } else {
    if (pad_mode == "VALID" || pad_mode == "valid") {
      if (std::any_of(pad_list.begin(), pad_list.end(), [](int i) { return i == 0; })) {
        has_pad = true;
      }
    }
  }
  if (!has_pad) {
    pad_list = {0, 0, 0, 0};
  }
  auto k_h = (kernel_size[0] - 1) * dilation[index2] + 1;
  auto k_w = (kernel_size[index1] - 1) * dilation[index3] + 1;
  auto out_h = (h + pad_list[0] + pad_list[index1] - k_h) / stride[index2] + 1;
  auto out_w = (w + pad_list[index2] + pad_list[index3] - k_w) / stride[index3] + 1;
  std::vector<int64_t> output = {n, out_h, out_w, out_channel};
  return output;
}

TypeId Conv2dOp::InferType(const NodePtrList &inputs, const DAttrs &attrs) {
  if (attrs.find("dst_type") == attrs.end()) return inputs[0]->type;
  auto dst_type = attrs.find("dst_type")->second;
  if (dst_type->isa<Type>()) {
    return dst_type->cast<TypePtr>()->type_id();
  }
  return StringToTypeId(GetValue<std::string>(dst_type));
}

DShape TransposeOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "perm");
  auto perm = GetListInt(attrs.find("perm")->second);
  auto &old_shape = inputs[0]->shape;
  DShape new_shape;
  if (perm.size() != old_shape.size()) {
    MS_LOG(EXCEPTION) << "perm.size() != old_shape.size(). " << perm.size() << " vs " << old_shape.size();
  }
  (void)std::transform(perm.begin(), perm.end(), std::back_inserter(new_shape),
                       [&old_shape](int64_t p) { return old_shape[LongToSize(p)]; });
  return new_shape;
}

DFormat TransposeOp::InferFormat(const NodePtrList &inputs, const DAttrs &attrs) {
  // only support NCHW/NHWC now
  if (inputs[0]->shape.size() != 4) return kOpFormat_DEFAULT;
  CHECK_ATTR(attrs, "perm");
  auto perm = GetListInt(attrs.find("perm")->second);
  const auto &ori_format = inputs[0]->format;
  if (ori_format == kOpFormat_DEFAULT || ori_format == kOpFormat_NCHW) {
    std::vector<int64_t> nchw2nhwc = {0, 2, 3, 1};
    if (perm == nchw2nhwc) return kOpFormat_NHWC;
  } else if (ori_format == kOpFormat_NHWC) {
    std::vector<int64_t> nhwc2nchw = {0, 3, 1, 2};
    if (perm == nhwc2nchw) return kOpFormat_DEFAULT;
  }
  return kOpFormat_DEFAULT;
}

DShape MatMulOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  std::vector<int64_t> shape0 = inputs[0]->shape;
  std::vector<int64_t> shape1 = inputs[1]->shape;
  if (shape0.size() != 2 || shape1.size() != 2) {
    MS_LOG(EXCEPTION) << "MatMul's input's dimension must be 2, but got " << shape0.size() << " and " << shape1.size();
  }
  CHECK_ATTR(attrs, "transpose_a");
  CHECK_ATTR(attrs, "transpose_b");
  auto transpose_a = GetValue<bool>(attrs.find("transpose_a")->second);
  auto transpose_b = GetValue<bool>(attrs.find("transpose_b")->second);
  int64_t m = transpose_a ? shape0[1] : shape0[0];
  int64_t k1 = transpose_a ? shape0[0] : shape0[1];
  int64_t k2 = transpose_b ? shape1[1] : shape1[0];
  int64_t n = transpose_b ? shape1[0] : shape1[1];
  if (k1 != k2) {
    MS_LOG(EXCEPTION) << "MatMul's inputs have different k value " << k1 << " vs " << k2;
  }
  std::vector<int64_t> output = {m, n};
  return output;
}

TypeId MatMulOp::InferType(const NodePtrList &inputs, const DAttrs &attrs) {
  if (attrs.find("dst_type") == attrs.end()) return inputs[0]->type;
  auto dst_type = attrs.find("dst_type")->second;
  if (dst_type->isa<Type>()) {
    return dst_type->cast<TypePtr>()->type_id();
  }
  return StringToTypeId(GetValue<std::string>(dst_type));
}

DShape PadAkgOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  std::vector<int64_t> shape0 = inputs[0]->shape;
  size_t n = shape0.size();
  CHECK_ATTR(attrs, "head");
  CHECK_ATTR(attrs, "tail");
  std::vector<int64_t> pad_before = GetListInt(attrs.find("head")->second);
  std::vector<int64_t> pad_after = GetListInt(attrs.find("tail")->second);
  if (pad_before.size() != n || pad_after.size() != n) {
    MS_LOG(EXCEPTION) << "Input dimension and pad mismatch: " << n << " vs " << pad_before.size() << " vs "
                      << pad_after.size();
  }
  std::vector<int64_t> output;
  for (size_t i = 0; i < n; i++) {
    (void)output.emplace_back(shape0[i] + pad_before[i] + pad_after[i]);
  }
  return output;
}

DShape UnPadAkgOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  std::vector<int64_t> shape0 = inputs[0]->shape;
  size_t n = shape0.size();
  CHECK_ATTR(attrs, "tail");
  std::vector<int64_t> unpad_after = GetListInt(attrs.find("tail")->second);
  if (unpad_after.size() != n) {
    MS_LOG(EXCEPTION) << "Input dimension and pad mismatch: " << n << " vs " << unpad_after.size();
  }
  std::vector<int64_t> output;
  for (size_t i = 0; i < n; i++) {
    (void)output.emplace_back(shape0[i] - unpad_after[i]);
  }
  return output;
}

void ComplexOp::CheckType(const NodePtrList &inputs, const DAttrs &) {
  if (inputs[0]->type != TypeId::kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "Complex's input[0] should be float32, but got " << TypeIdToString(inputs[0]->type, true);
  }
  if (inputs[0]->type != inputs[1]->type) {
    MS_LOG(EXCEPTION) << "Complex's input[0] and inputs[1]'s type mismatch: " << TypeIdToString(inputs[0]->type, true)
                      << " vs " << TypeIdToString(inputs[1]->type, true);
  }
}

DShape StandardNormalOp::InferShape(const NodePtrList &, const DAttrs &attrs) {
  CHECK_ATTR(attrs, "shape");
  return GetListInt(attrs.find("shape")->second);
}
}  // namespace luojianet_ms::graphkernel::inner
