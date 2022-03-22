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

#include "graph/passes/infer_value_range_pass.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "graph/passes/constant_folding_pass.h"
#include "graph/utils/type_utils.h"
#include "common/ge/ge_util.h"

using std::unique_ptr;
namespace ge {
namespace {
#define GET_DATA_BY_DTYPE(DTYPE, TYPE)                                                                  \
  case (DTYPE):                                                                                         \
    ConstructValueRange<TYPE>(lower_boundary_tensor, upper_boundary_tensor, output_tensor_value_range); \
    break;

void SerialShapeRange(const GeTensorDescPtr &desc, std::string &desc_str) {
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)desc->GetShapeRange(shape_range);
  desc_str += formats::RangeToString(shape_range);
  shape_range.clear();
  (void)desc->GetOriginShapeRange(shape_range);
  desc_str += ",";
  desc_str += formats::RangeToString(shape_range);
  shape_range.clear();
}

Status RunCpuKernelForValueRange(NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                 std::vector<GeTensorPtr> &outputs) {
  // RunOpKernelWithCheck, RunOpKernel for test
  auto ret = ConstantFoldingPass::RunOpKernel(node, inputs, outputs);
  if (ret != SUCCESS) {
    auto op_kernel = folding_pass::GetKernelByType(node);
    if (op_kernel == nullptr) {
      GELOGW("Calculate value range failed, no op kernel for node %s type %s", node->GetName().c_str(),
             node->GetType().c_str());
      return NOT_CHANGED;
    }

    ret = op_kernel->Compute(node->GetOpDesc(), inputs, outputs);
    if (ret != SUCCESS) {
      GELOGW("Calculate value range failed, node %s run cpu kernel failed.", node->GetName().c_str());
      return NOT_CHANGED;
    }
  }
  GELOGI("Node %s type %s, run cpu kernel success.", node->GetName().c_str(), node->GetType().c_str());
  return SUCCESS;
}
}  // namespace

graphStatus InferValueRangePass::Infer(NodePtr &node) {
  auto infer_value_range_param = OperatorFactoryImpl::GetInferValueRangePara(node->GetType());

  // Use registered func to calculate value range
  if (!infer_value_range_param.use_cpu_kernel) {
    if (infer_value_range_param.infer_value_func == nullptr) {
      GELOGW("The registered func of node %s to infer value range is nullptr.", node->GetName().c_str());
      return GRAPH_NOT_CHANGED;
    }
    Operator op = OpDescUtils::CreateOperatorFromNode(node);
    auto ret = node->GetOpDesc()->CallInferValueRangeFunc(op);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("Node %s call infer value range func failed, ret: %u.", node->GetName().c_str(), ret);
      return GRAPH_NOT_CHANGED;
    }
    GELOGD("Node %s infer value range func succeed by registered func.", node->GetName().c_str());
    return GRAPH_SUCCESS;
  }

  // Deal with scenes with unknown value range
  bool has_unknown_value_range = false;
  bool has_zero_in_value_range = false;
  CheckInputValueRange(node, has_unknown_value_range, has_zero_in_value_range);
  if (has_unknown_value_range) {
    if (has_zero_in_value_range) {
      // When there is zero in input value range, it is unreasonable to always set output value range {1:-1}.
      GELOGW("Node %s has -1 and 0 in value range, skip setting value range.", node->GetName().c_str());
      return GRAPH_NOT_CHANGED;
    }
    GELOGI("Node %s has unknown value range in input tensors, set value range {1:-1}, and skip cpu kernel.",
           node->GetName().c_str());
    return GenerateWorstValueRange(node);
  }

  // Use CPU kernel func to calculate value range
  auto ret = ConstructInputAndInferValueRange(node);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("Use CPU kernel to calculate value range failed. node: %s, ret: %u", node->GetName().c_str(), ret);
    return GRAPH_NOT_CHANGED;
  }
  GELOGD("Node %s infer value range func succeed by running cpu kernel.", node->GetName().c_str());
  return GRAPH_SUCCESS;
}

std::string InferValueRangePass::SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const {
  std::stringstream ss;
  ss << "[";
  ss << "(shape:[" << tensor_desc->MutableShape().ToString() << "]),";
  string range_str;
  SerialShapeRange(tensor_desc, range_str);
  ss << "(shape_range:" << range_str << "),";
  std::vector<std::pair<int64_t, int64_t>> value_range;
  (void)tensor_desc->GetValueRange(value_range);
  string value_range_str = formats::RangeToString(value_range);
  ss << "(value_range:" << value_range_str << ")]";
  return ss.str();
}

bool InferValueRangePass::NeedInfer(const NodePtr &node) const {
  auto infer_value_range_param = OperatorFactoryImpl::GetInferValueRangePara(node->GetType());
  if (!infer_value_range_param.is_initialized) {
    GELOGD("Node %s does not register func to infer value range, skip infer_value_range_pass.",
           node->GetName().c_str());
    return false;
  }

  if (infer_value_range_param.when_call == INPUT_IS_DYNAMIC) {
    // Only do infer for node that all inputs are dynamic, such as shape
    if (InputIsDynamic(node)) {
      return true;
    }
    GELOGD("Node %s register func to infer value range and when_call is INPUT_IS_DYNAMIC, but check input failed.",
           node->GetName().c_str());
  } else if (infer_value_range_param.when_call == INPUT_HAS_VALUE_RANGE) {
    // Only do infer for node that all inputs have value_range or node type of inputs is constant/const
    if (InputIsConstOrHasValueRange(node)) {
      return true;
    }
    GELOGD("Node %s register func to infer value range and when_call is INPUT_HAS_VALUE_RANGE, but check input failed.",
           node->GetName().c_str());
  }
  GELOGD("Node %s does not need to infer value range, skip infer_value_range_pass.", node->GetName().c_str());
  return false;
}

bool InferValueRangePass::InputIsDynamic(const NodePtr &node) const{
  bool input_is_dynamic = false;
  auto cur_op_desc = node->GetOpDesc();
  for (const auto &input_desc : cur_op_desc->GetAllInputsDescPtr()) {
    auto dims = input_desc->GetShape().GetDims();
    for (auto dim : dims) {
      if (dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM) {
        input_is_dynamic = true;
        break;
      }
    }
  }
  return input_is_dynamic;
}

bool InferValueRangePass::InputIsConstOrHasValueRange(const NodePtr &node) const {
  bool input_is_const_or_has_value_range = true;
  auto cur_op_desc = node->GetOpDesc();
  auto in_data_anchors = node->GetAllInDataAnchors();
  for (size_t i = 0; i < in_data_anchors.size(); ++i) {
    auto peer_out_anchor = in_data_anchors.at(i)->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto peer_node = peer_out_anchor->GetOwnerNode();
    if (peer_node == nullptr || peer_node->GetOpDesc() == nullptr) {
      continue;
    }
    if ((peer_node->GetType() == CONSTANT) || (peer_node->GetType() == CONSTANTOP)) {
      continue;
    }

    const auto &input_desc = cur_op_desc->GetInputDesc(i);
    std::vector<std::pair<int64_t, int64_t>> value_range;
    (void)input_desc.GetValueRange(value_range);
    if (value_range.empty()) {
      GELOGD("Node %s input %zu does not have value range, skip infer_value_range_pass for current node.",
             node->GetName().c_str(), i);
      input_is_const_or_has_value_range = false;
      break;
    }
  }
  return input_is_const_or_has_value_range;
}

void InferValueRangePass::CheckInputValueRange(const NodePtr &node, bool &has_unknown_value_range,
                                               bool &has_zero_in_value_range) const {
  has_unknown_value_range = false;
  has_zero_in_value_range = false;
  auto cur_op_desc = node->GetOpDesc();
  for (const auto &input_desc : cur_op_desc->GetAllInputsDescPtr()) {
    std::vector<std::pair<int64_t, int64_t>> input_desc_value_range;
    input_desc->GetValueRange(input_desc_value_range);
    if (!input_desc_value_range.empty()) {
      for (const auto &range : input_desc_value_range) {
        if (range.first == 0 || range.second == 0) {
          GELOGD("Node %s input tensors have zero in value range %s.", node->GetName().c_str(),
                 formats::RangeToString(input_desc_value_range).c_str());
          has_zero_in_value_range = true;
        }
        if (range.first == -1 || range.second == -1) {
          GELOGD("Node %s input tensors have unknown value range, value range is %s.", node->GetName().c_str(),
                 formats::RangeToString(input_desc_value_range).c_str());
          has_unknown_value_range = true;
        }
      }
    }
  }
}

graphStatus InferValueRangePass::UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) {
  if (src == nullptr || dst == nullptr) {
    REPORT_CALL_ERROR("E19999", "While updating tensor desc, input desc is null.");
    GELOGE(GRAPH_FAILED, "[Param][check] While updating tensor desc, input desc is null.");
    return GRAPH_FAILED;
  }

  changed = false;
  std::vector<std::pair<int64_t, int64_t>> src_value_range;
  std::vector<std::pair<int64_t, int64_t>> dst_value_range;
  (void)src->GetValueRange(src_value_range);
  (void)dst->GetValueRange(dst_value_range);
  if (src_value_range != dst_value_range) {
    GELOGD("While updating tensor desc, value range has been changed, src value range: %s, dst value range: %s.",
           formats::RangeToString(src_value_range).c_str(), formats::RangeToString(dst_value_range).c_str());
    changed = true;
  }

  dst->SetValueRange(src_value_range);
  return GRAPH_SUCCESS;
}

graphStatus InferValueRangePass::UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src,
                                                           GeTensorDescPtr &dst) {
  std::vector<std::pair<int64_t, int64_t>> ref_out_tensor_value_range;
  auto ref_out_tensor = src.at(0);
  (void)ref_out_tensor->GetValueRange(ref_out_tensor_value_range);
  for (auto &ref_tensor : src) {
    std::vector<std::pair<int64_t, int64_t>> ref_tensor_value_range;
    (void)ref_tensor->GetValueRange(ref_tensor_value_range);

    if (ref_tensor_value_range.size() != ref_out_tensor_value_range.size()) {
      GELOGD("Update TensorDesc %s failed, rank of value ranges %s and %s are not the same, skip value range refresh.",
             dst->GetName().c_str(), formats::RangeToString(ref_out_tensor_value_range).c_str(),
             formats::RangeToString(ref_tensor_value_range).c_str());
      return GRAPH_SUCCESS;
    }

    for (size_t j = 0; j < ref_out_tensor_value_range.size(); j++) {
      if ((ref_out_tensor_value_range.at(j).first != ref_tensor_value_range.at(j).first) ||
          (ref_out_tensor_value_range.at(j).second != ref_tensor_value_range.at(j).second)) {
        ref_out_tensor_value_range[j] = std::make_pair(1, -1);
      }
    }
  }
  GELOGD("While updating output desc from subgraphs, set parent node desc value range %s.",
         formats::RangeToString(ref_out_tensor_value_range).c_str());
  dst->SetValueRange(ref_out_tensor_value_range);
  return GRAPH_SUCCESS;
}

graphStatus InferValueRangePass::UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                                       GeTensorDescPtr &dst) {
  REPORT_INNER_ERROR("E19999",
                     "Update TensorDesc %s failed. In dynamic multi-dims size scene, there should be no value range.",
                     dst->GetName().c_str());
  GELOGE(GRAPH_FAILED,
         "[Update][TensorDesc] %s failed. In dynamic multi-dims size scene, there should be no value range.",
         dst->GetName().c_str());
  return GRAPH_FAILED;
}

graphStatus InferValueRangePass::GenerateWorstValueRange(NodePtr &node) {
  GELOGI("Node %s does not run cpu kernel, because input value range has -1.", node->GetName().c_str());
  OpDescPtr op_desc = node->GetOpDesc();
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    auto output_desc = op_desc->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      continue;
    }
    auto output_i_shape = output_desc->GetShape();
    auto output_i_shape_size = output_i_shape.GetShapeSize();
    if (output_i_shape_size < 0) {
      GELOGD("Node %s output shape is unknown, cannot infer value range, shape is %s.", node->GetName().c_str(),
             formats::ShapeToString(output_i_shape).c_str());
      return GRAPH_NOT_CHANGED;
    }

    std::vector<std::pair<int64_t, int64_t>> output_i_value_range(output_i_shape_size, {1, -1});
    if (output_i_shape.IsScalar()) {
      output_i_value_range.emplace_back(1, -1);
    }
    output_desc->SetValueRange(output_i_value_range);
    GELOGD("Node %s output %zu shape is %s, the generated worst value range is %s.", node->GetName().c_str(), i,
           formats::ShapeToString(output_i_shape).c_str(), formats::RangeToString(output_i_value_range).c_str());
  }
  return GRAPH_SUCCESS;
}

template <typename T>
graphStatus InferValueRangePass::ConstructData(const GeTensorDesc &tensor_desc, bool use_floor_value,
                                               GeTensorPtr &output_ptr) {
  std::vector<std::pair<int64_t, int64_t>> value_range;
  (void)tensor_desc.GetValueRange(value_range);
  size_t value_range_data_num = value_range.size();
  auto tensor_shape = tensor_desc.GetShape();
  bool value_range_and_tensor_shape_matched = true;
  if (tensor_shape.IsScalar()){
    // scalar tensor has only one value_range pair
    if (value_range_data_num != 1) {
      value_range_and_tensor_shape_matched = false;
    }
  } else {
    // normal tensor, value_range size is equal to tensor shape size.
    if (static_cast<int64_t>(value_range_data_num) != tensor_shape.GetShapeSize()) {
      value_range_and_tensor_shape_matched = false;
    }
  }
  if (!value_range_and_tensor_shape_matched) {
    GELOGW("Input %s value range and tensor shape do not match. Value range size is %zu, tensor shape is %s.",
           tensor_desc.GetName().c_str(), value_range_data_num, formats::ShapeToString(tensor_shape).c_str());
    return GRAPH_PARAM_INVALID;
  }

  unique_ptr<T[]> buf(new (std::nothrow) T[value_range_data_num]());
  if (buf == nullptr) {
    REPORT_INNER_ERROR("E19999", "New buf failed");
    GELOGE(MEMALLOC_FAILED, "New buf failed");
    return GRAPH_FAILED;
  }
  for (size_t j = 0; j < value_range_data_num; ++j) {
    auto value_range_j = use_floor_value ? value_range[j].first : value_range[j].second;
    buf[j] = static_cast<T>(value_range_j);
  }

  if (output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), value_range_data_num * sizeof(T)) != GRAPH_SUCCESS) {
    GELOGW("Set data failed while constructing value range input tensor.");
    return GRAPH_NOT_CHANGED;
  }
  return GRAPH_SUCCESS;
}

graphStatus InferValueRangePass::ConstructDataByType(const GeTensorDesc &tensor_desc, bool use_floor_value,
                                                     GeTensorPtr &output_ptr) {
  graphStatus ret = GRAPH_SUCCESS;
  auto data_type = tensor_desc.GetDataType();
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  switch (data_type) {
    case DT_FLOAT:
      ret = ConstructData<float>(tensor_desc, use_floor_value, output_ptr);
      break;
    case DT_DOUBLE:
      ret = ConstructData<double>(tensor_desc, use_floor_value, output_ptr);
      break;
    case DT_UINT8:
      ret = ConstructData<uint8_t>(tensor_desc, use_floor_value, output_ptr);
      break;
    case DT_INT8:
      ret = ConstructData<int8_t>(tensor_desc, use_floor_value, output_ptr);
      break;
    case DT_UINT16:
      ret = ConstructData<uint16_t>(tensor_desc, use_floor_value, output_ptr);
      break;
    case DT_INT16:
      ret = ConstructData<int16_t>(tensor_desc, use_floor_value, output_ptr);
      break;
    case DT_INT32:
      ret = ConstructData<int32_t>(tensor_desc, use_floor_value, output_ptr);
      break;
    case DT_INT64:
      ret = ConstructData<int64_t>(tensor_desc, use_floor_value, output_ptr);
      break;
    default:
      GELOGW("Data type:%s is not supported.", TypeUtils::DataTypeToSerialString(data_type).c_str());
      ret = GRAPH_PARAM_INVALID;
  }
  return ret;
}

vector<ConstGeTensorPtr> InferValueRangePass::ConstructInputTensors(const NodePtr &node, bool use_floor_value) {
  vector<ConstGeTensorPtr> input_tensors;
  auto cur_op_desc = node->GetOpDesc();
  auto in_data_anchors = node->GetAllInDataAnchors();
  for (size_t i = 0; i < in_data_anchors.size(); ++i) {
    auto peer_out_anchor = in_data_anchors.at(i)->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto peer_node = peer_out_anchor->GetOwnerNode();
    if (peer_node == nullptr) {
      continue;
    }

    // construct input tensor by constant node
    if ((peer_node->GetType() == CONSTANT) || (peer_node->GetType() == CONSTANTOP)) {
      vector<GeTensorPtr> const_weight = OpDescUtils::MutableWeights(peer_node);
      if (const_weight.empty()) {
        GELOGW("MutableWeights failed, weight is empty, node: %s(%s)", peer_node->GetName().c_str(),
               peer_node->GetType().c_str());
        return vector<ConstGeTensorPtr>();
      }
      // const/constant op has only one weight
      if (const_weight.at(0) == nullptr) {
        GELOGW("MutableWeights failed, weight of constant is null, node name: %s(%s)",
               peer_node->GetName().c_str(), peer_node->GetType().c_str());
        return vector<ConstGeTensorPtr>();
      }
      input_tensors.push_back(const_weight.at(0));
      GELOGD("Node %s construct input tensor %zu by constant node.", node->GetName().c_str(), input_tensors.size());
      continue;
    }

    // construct input tensor by boundary of value range
    const auto &input_tensor_desc = cur_op_desc->GetInputDesc(i);
    GeTensorPtr tmp_tensor_ptr = MakeShared<GeTensor>(input_tensor_desc);
    if (tmp_tensor_ptr == nullptr) {
      REPORT_INNER_ERROR("E19999", "Make shared failed");
      GELOGE(MEMALLOC_FAILED, "Make shared failed");
      return vector<ConstGeTensorPtr>();
    }

    auto ret = ConstructDataByType(input_tensor_desc, use_floor_value, tmp_tensor_ptr);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("Construct input tensor by boundary of value range failed for input %s.",
             input_tensor_desc.GetName().c_str());
      return vector<ConstGeTensorPtr>();
    }
    input_tensors.push_back(tmp_tensor_ptr);
    GELOGD("Node %s construct input tensor %zu by input desc value range.", node->GetName().c_str(),
           input_tensors.size());
  }

  return input_tensors;
}

graphStatus InferValueRangePass::ConstructInputAndInferValueRange(NodePtr &node) {
  auto inputs = ConstructInputTensors(node, true);
  if (inputs.empty()) {
    return GRAPH_PARAM_INVALID;
  }
  vector<GeTensorPtr> lower_boundary_outputs;
  auto ret = RunCpuKernelForValueRange(node, inputs, lower_boundary_outputs);
  if (ret != SUCCESS) {
    GELOGW("Node %s run cpu kernel failed while calculating value range.", node->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }

  inputs = ConstructInputTensors(node, false);
  if (inputs.empty()) {
    return GRAPH_PARAM_INVALID;
  }
  vector<GeTensorPtr> upper_boundary_outputs;
  ret = RunCpuKernelForValueRange(node, inputs, upper_boundary_outputs);
  if (ret != SUCCESS) {
    GELOGW("Node %s run cpu kernel failed while calculating value range.", node->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }

  // construct value range from output tensor
  OpDescPtr node_desc = node->GetOpDesc();
  std::vector<std::pair<int64_t, int64_t>> output_tensor_value_range;
  size_t node_output_desc_size = node_desc->GetOutputsSize();
  for (size_t i = 0; i < node_output_desc_size; ++i) {
    output_tensor_value_range.clear();
    auto output_tensor_desc = node_desc->MutableOutputDesc(i);
    auto output_shape_size = output_tensor_desc->GetShape().GetShapeSize();
    auto lower_boundary_tensor = lower_boundary_outputs[i];
    auto lower_boundary_shape = lower_boundary_tensor->GetTensorDesc().GetShape();
    auto upper_boundary_tensor = upper_boundary_outputs[i];
    auto upper_boundary_shape = upper_boundary_tensor->GetTensorDesc().GetShape();
    if (lower_boundary_shape.GetShapeSize() != output_shape_size ||
        upper_boundary_shape.GetShapeSize() != output_shape_size) {
      GELOGD(
        "Cpu kernel result shapes %s, %s and output shape %s do not match, can not infer value range for output %s.",
        formats::ShapeToString(lower_boundary_shape).c_str(), formats::ShapeToString(upper_boundary_shape).c_str(),
        formats::ShapeToString(output_tensor_desc->GetShape()).c_str(), output_tensor_desc->GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }

    auto data_type = output_tensor_desc->GetDataType();
    switch (data_type) {
      GET_DATA_BY_DTYPE(DT_INT8, int8_t)
      GET_DATA_BY_DTYPE(DT_INT16, int16_t)
      GET_DATA_BY_DTYPE(DT_INT32, int32_t)
      GET_DATA_BY_DTYPE(DT_INT64, int64_t)
      GET_DATA_BY_DTYPE(DT_UINT8, uint8_t)
      GET_DATA_BY_DTYPE(DT_UINT16, uint16_t)
      GET_DATA_BY_DTYPE(DT_UINT32, uint32_t)
      GET_DATA_BY_DTYPE(DT_UINT64, uint64_t)
      GET_DATA_BY_DTYPE(DT_FLOAT, float)
      GET_DATA_BY_DTYPE(DT_DOUBLE, double)
      default:
        GELOGW("Data type:%s is not supported.", TypeUtils::DataTypeToSerialString(data_type).c_str());
        return GRAPH_PARAM_INVALID;
    }
    output_tensor_desc->SetValueRange(output_tensor_value_range);
    GELOGD("Node %s calculates output %zu value range %s by running cpu kernel.", node->GetName().c_str(), i,
           formats::RangeToString(output_tensor_value_range).c_str());
  }
  return GRAPH_SUCCESS;
}

template <typename T>
void InferValueRangePass::ConstructValueRange(const GeTensorPtr &left_tensor, const GeTensorPtr &right_tensor,
                                              std::vector<std::pair<int64_t, int64_t>> &value_range) {
  auto x = reinterpret_cast<const T *>(left_tensor->GetData().GetData());
  auto y = reinterpret_cast<const T *>(right_tensor->GetData().GetData());
  if (x == nullptr || y == nullptr) {
    GELOGI("Output tensor of cpu kernel does not have data, no way to set value range.");
    return;
  }
  auto left_tensor_shape = left_tensor->GetTensorDesc().GetShape();
  for (auto j = 0; j < left_tensor_shape.GetShapeSize(); ++j) {
    auto left = static_cast<int64_t>(*(x + j));
    auto right = static_cast<int64_t>(*(y + j));
    value_range.emplace_back(left, right);
  }

  if (left_tensor_shape.IsScalar()) {
    GELOGD("When inferring value range, output tensors of cpu kernel are scalar tensors.");
    value_range.emplace_back(static_cast<int64_t>(*x), static_cast<int64_t>(*y));
  }
}
}  // namespace ge
