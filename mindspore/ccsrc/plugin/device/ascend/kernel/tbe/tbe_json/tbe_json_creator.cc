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

#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_creator.h"
#include <memory>
#include <map>
#include <utility>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_adapter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynaminc_shape_util.h"
#include "utils/ms_context.h"
#include "runtime/dev.h"
#include "utils/ms_utils.h"
#include "include/common/utils/json_operation_utils.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"

namespace mindspore::kernel {
namespace {
static std::unordered_map<std::string, ATTR_DTYPE> type_attr_dtype_map = {
  {kVTypeInt, ATTR_DTYPE::ATTR_INT32},
  {kVTypeInt64, ATTR_DTYPE::ATTR_INT64},
  {kVTypeStr, ATTR_DTYPE::ATTR_STR},
  {kVTypeBool, ATTR_DTYPE::ATTR_BOOL},
  {kVTypeFloat, ATTR_DTYPE::ATTR_FLOAT32},
  {kVTypeListInt, ATTR_DTYPE::ATTR_LIST_INT32},
  {kVTypeListFloat, ATTR_DTYPE::ATTR_LIST_FLOAT32},
  {kVTypeListUInt64, ATTR_DTYPE::ATTR_LIST_UINT64},
  {kVTypeListListInt, ATTR_DTYPE::ATTR_LIST_LIST_INT64}};

static std::map<ATTR_DTYPE, std::string> tbe_attr_dtype_to_string_map = {
  {ATTR_INT8, "int8"},
  {ATTR_UINT8, "uint8"},
  {ATTR_INT16, "int16"},
  {ATTR_UINT16, "uint16"},
  {ATTR_INT32, "int32"},
  {ATTR_UINT32, "uint32"},
  {ATTR_INT64, "int64"},
  {ATTR_UINT64, "uint64"},
  {ATTR_FLOAT32, "float32"},
  {ATTR_DOUBLE, "double"},
  {ATTR_BOOL, "bool"},
  {ATTR_STR, "str"},
  {ATTR_LIST_INT8, "list_int8"},
  {ATTR_LIST_UINT8, "list_uint8"},
  {ATTR_LIST_INT16, "list_int16"},
  {ATTR_LIST_UINT16, "list_uint16"},
  {ATTR_LIST_INT32, "list_int32"},
  {ATTR_LIST_UINT32, "list_uint32"},
  {ATTR_LIST_INT64, "list_int64"},
  {ATTR_LIST_UINT64, "list_uint64"},
  {ATTR_LIST_FLOAT32, "list_float32"},
  {ATTR_LIST_DOUBLE, "list_double"},
  {ATTR_LIST_BOOL, "list_bool"},
  {ATTR_LIST_STR, "list_str"},
  {ATTR_LIST_LIST_INT64, "list_list_int64"},
  {ATTR_LIST_LIST_FLOAT, "list_list_float"},
};

bool ParseListIntValue(const mindspore::ValuePtr &value, std::vector<int64_t> *attr_value) {
  auto value_type = value->type();
  if (value_type == nullptr) {
    MS_LOG(ERROR) << "Value's type is null.";
    return false;
  }
  if (value_type->ToString() == kVTypeInt64) {
    attr_value->push_back(GetValue<int64_t>(value));
  } else {
    auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
    if (!vec.empty()) {
      if (vec[0]->isa<Int32Imm>()) {
        std::vector<int32_t> attr_value_me = GetValue<std::vector<int32_t>>(value);
        (void)std::transform(attr_value_me.begin(), attr_value_me.end(), std::back_inserter(*attr_value),
                             [](const int &value) { return static_cast<int64_t>(value); });
      } else {
        *attr_value = GetValue<std::vector<int64_t>>(value);
      }
    }
  }
  return true;
}

bool ParseAttrValue(const std::string &type, const mindspore::ValuePtr &value, nlohmann::json *attr_obj) {
  MS_EXCEPTION_IF_NULL(attr_obj);
  if (value == nullptr) {
    MS_LOG(ERROR) << "Node's attr value is null.";
    return false;
  }
  auto result = type_attr_dtype_map.find(type);
  if (result == type_attr_dtype_map.end()) {
    MS_LOG(ERROR) << "Type: " << type << "not support";
    return false;
  }

  auto dtype_string = tbe_attr_dtype_to_string_map.find(result->second);
  if (dtype_string == tbe_attr_dtype_to_string_map.end()) {
    MS_LOG(ERROR) << "Can't convert attr dtype " << result->second << " to string";
    return false;
  }
  (*attr_obj)[kJDtype] = dtype_string->second;

  switch (result->second) {
    case ATTR_DTYPE::ATTR_INT32:
      (*attr_obj)[kJValue] = value->isa<Int32Imm>() ? GetValue<int>(value) : GetValue<int64_t>(value);
      break;
    case ATTR_DTYPE::ATTR_INT64:
      (*attr_obj)[kJValue] = GetValue<int64_t>(value);
      break;
    case ATTR_DTYPE::ATTR_STR: {
      auto attr_value = GetValue<std::string>(value);
      (*attr_obj)[kJValue] = attr_value == kOpFormat_FRAC_Z ? kJOpFormat_FRACTAL_Z : attr_value;
      break;
    }
    case ATTR_DTYPE::ATTR_BOOL:
      (*attr_obj)[kJValue] = GetValue<bool>(value);
      break;
    case ATTR_DTYPE::ATTR_FLOAT32:
      (*attr_obj)[kJValue] = GetValue<float>(value);
      break;
    case ATTR_DTYPE::ATTR_LIST_INT32: {
      std::vector<int64_t> attr_value;
      if (!ParseListIntValue(value, &attr_value)) {
        MS_LOG(ERROR) << "Parse list_value failed, maybe the input is a nullptr.";
        return false;
      }
      (*attr_obj)[kJValue] = attr_value;
      break;
    }
    case ATTR_DTYPE::ATTR_LIST_FLOAT32: {
      auto value_type = value->type();
      if (value_type == nullptr) {
        MS_LOG(ERROR) << "Value's type is null.";
        return false;
      }
      (*attr_obj)[kJValue] = value_type->ToString() == kVTypeFloat ? std::vector<float>{GetValue<float>(value)}
                                                                   : GetValue<std::vector<float>>(value);
      break;
    }
    case ATTR_DTYPE::ATTR_LIST_UINT64:
      (*attr_obj)[kJValue] = GetValue<std::vector<size_t>>(value);
      break;
    case ATTR_DTYPE::ATTR_LIST_LIST_INT64:
      (*attr_obj)[kJValue] = GetValue<std::vector<std::vector<int64_t>>>(value);
      break;

    default:
      MS_LOG(ERROR) << "Type: " << type << "not support";
      return false;
  }
  return true;
}

bool ParseAttrDefaultValue(const std::string &type, const std::string &value, nlohmann::json *attr_obj) {
  MS_EXCEPTION_IF_NULL(attr_obj);
  auto result = type_attr_dtype_map.find(type);
  if (result == type_attr_dtype_map.end()) {
    MS_LOG(ERROR) << "Type: " << type << "not support";
    return false;
  }

  auto dtype_string = tbe_attr_dtype_to_string_map.find(result->second);
  if (dtype_string == tbe_attr_dtype_to_string_map.end()) {
    MS_LOG(ERROR) << "Can't convert attr dtype " << result->second << " to string";
    return false;
  }
  (*attr_obj)[kJDtype] = dtype_string->second;

  switch (result->second) {
    case ATTR_DTYPE::ATTR_INT32:
      (*attr_obj)[kJValue] = std::stoi(value);
      break;
    case ATTR_DTYPE::ATTR_INT64:
      (*attr_obj)[kJValue] = std::stoll(value);
      break;
    case ATTR_DTYPE::ATTR_STR:
      (*attr_obj)[kJValue] = value;
      break;
    case ATTR_DTYPE::ATTR_BOOL: {
      bool attr_value = false;
      std::istringstream(value) >> std::boolalpha >> attr_value;
      (*attr_obj)[kJValue] = attr_value;
      break;
    }
    case ATTR_DTYPE::ATTR_FLOAT32:
      (*attr_obj)[kJValue] = std::stof(value);
      break;
    case ATTR_DTYPE::ATTR_LIST_INT32: {
      std::stringstream string_value(value);
      std::string list_elem;
      std::vector<int64_t> attrs_value;
      while (std::getline(string_value, list_elem, ',')) {
        attrs_value.push_back(std::stoi(list_elem));
      }
      (*attr_obj)[kJValue] = attrs_value;
      break;
    }
    default:
      MS_LOG(ERROR) << "Type: " << type << "not support";
      return false;
  }
  return true;
}
}  // namespace

bool TbeJsonCreator::GenComputeJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  MS_LOG(DEBUG) << "Start.";

  if (!GenInputsJson(anf_node, compute_json)) {
    MS_LOG(ERROR) << "generate inputs json failed, node full name:" << anf_node->fullname_with_scope();
    return false;
  }
  if (!GenOutputsJson(anf_node, compute_json)) {
    MS_LOG(ERROR) << "generate outputs json failed, node full name:" << anf_node->fullname_with_scope();
    return false;
  }
  GenOutputDataDescJson(anf_node, compute_json);
  GenAttrsDescJson(anf_node, compute_json);
  GenComputeCommonJson(anf_node, compute_json);
  GenOtherJson(anf_node, compute_json);
  MS_LOG(DEBUG) << "End.";
  return true;
}

void TbeJsonCreator::GenFusionOpName(nlohmann::json *kernel_json, std::string prefix) {
  json_name_.clear();
  json_hash_ = GenJsonHash((*kernel_json));
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  json_name_ = std::move(prefix);
  auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  for (auto node_json : (*kernel_json)[kJOpList]) {
    if (GetJsonValue<std::string>(node_json, kJType) != kJData) {
      json_name_.append(node_json[kJFuncName]);
      json_name_.append("_");
    }
  }
  json_name_ = json_name_ + std::to_string(json_hash_) + "_" + std::to_string(device_id);
  MS_LOG(DEBUG) << "Generate Json name: " << json_name_;
  (*kernel_json)[kJFusionOpName] = json_name_;
}

void TbeJsonCreator::DeleteDescName(nlohmann::json *desc_jsons) {
  for (auto &desc_json : (*desc_jsons)) {
    if (desc_json.is_array()) {
      for (auto &desc_item : desc_json) {
        (void)desc_item.erase(kJName);
      }
    } else {
      (void)desc_json.erase(kJName);
    }
  }
}

size_t TbeJsonCreator::GenJsonHash(nlohmann::json tbe_json) {
  auto &op_lists = tbe_json.at(kJOpList);
  for (auto &op : op_lists) {
    (void)op.erase(kJName);
    (void)op.erase(kJOriName);
    (void)op.erase(kJPattern);
    (void)op.erase(kJOutputDataDesc);
    DeleteDescName(&op.at(kJOutputDesc));
    if (op[kJType] != kJData) {
      DeleteDescName(&op.at(kJInputDesc));
    }
  }
  return std::hash<std::string>()(op_lists.dump());
}

void TbeJsonCreator::AddOpNameForComputeNode(nlohmann::json *kernel_json) {
  auto op_name = GetJsonValue<std::string>((*kernel_json), kJFusionOpName);
  for (auto &node_json : (*kernel_json).at(kJOpList)) {
    // compute node
    if (GetJsonValue<std::string>(node_json, kJType) != kJData) {
      node_json[kJOpName] = op_name;
    }
  }
}

void TbeJsonCreator::GenAttrsJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(attrs_json);
  auto attrs_ptr = op_info->attrs_ptr();
  AttrsJsonPreProcessing(anf_node, &attrs_ptr, attrs_json);

  std::string op_name = common::AnfAlgo::GetCNodeName(anf_node);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &attr_ptr : attrs_ptr) {
    std::string attr_name = attr_ptr->name();
    nlohmann::json attr_obj;
    attr_obj[kJName] = attr_name;
    if (primitive->GetAttr(attr_name) != nullptr) {
      if (!ParseAttrValue(attr_ptr->type(), primitive->GetAttr(attr_name), &attr_obj)) {
        MS_LOG(EXCEPTION) << "op [ " << op_info->op_name() << " ]'s attr [ " << attr_name << " ] generates failed";
      }
      attr_obj[kJValid] = true;
    } else {
      auto default_value = attr_ptr->default_value();
      if (!default_value.empty()) {
        if (!ParseAttrDefaultValue(attr_ptr->type(), default_value, &attr_obj)) {
          MS_LOG(EXCEPTION) << "op [ " << op_info->op_name() << " ]'s default attr [ " << attr_name
                            << " ] generates failed";
        }
        attr_obj[kJValid] = true;
      } else {
        MS_LOG(INFO) << "op " << op_name << "'s attr \"" << attr_name << "\" should have a default value.";
        if (!op_info->impl_path().empty() && attr_ptr->param_type() == kJParamRequred) {
          MS_LOG(EXCEPTION) << "Op name: " << op_info->op_name() << " attr: " << attr_name
                            << " is required, but not set.";
        } else {
          attr_obj[kJValid] = false;
        }
      }
    }
    (*attrs_json).push_back(attr_obj);
  }

  if (!AttrsJsonPostProcessing(anf_node, op_info, attrs_json)) {
    MS_LOG(EXCEPTION) << "PostProcessing node attr error, node: " << anf_node->fullname_with_scope();
  }
}

void TbeJsonCreator::GenAttrsDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
  nlohmann::json attrs_json;
  GenAttrsJson(cnode, op_info_ptr, &attrs_json);
  if (!attrs_json.empty()) {
    (*compute_json)[kJAttrs] = attrs_json;
  }

  nlohmann::json attrs_desc;
  for (const auto &attr : attrs_json) {
    if (GetJsonValue<std::string>(attr, kJName) != kJIsRef && GetJsonValue<bool>(attr, kJValid)) {
      attrs_desc.push_back(attr.at(kJValue));
    }
  }
  if (!attrs_desc.empty()) {
    (*compute_json)[kJAttrDesc] = attrs_desc;
  }
}

void TbeJsonCreator::GenComputeCommonJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
  auto func_name = op_info_ptr->kernel_name();
  (*compute_json)[kJFuncName] = func_name;
  auto python_module_path = op_info_ptr->impl_path();
  if (python_module_path.empty()) {
    python_module_path = kPyPath;
  }

  auto iter = tbe::opTypeAdapter.find(op_name);
  (*compute_json)[kJType] = (iter != tbe::opTypeAdapter.end()) ? iter->second : op_name;
  (*compute_json)[kJPyModulePath] = python_module_path;
  (*compute_json)[kJDynamicCompileStatic] = op_info_ptr->dynamic_compile_static();
  (*compute_json)[kJInt64Mode] = false;
  (*compute_json)[kJName] = cnode->fullname_with_scope();
  (*compute_json)[kJPattern] = kernel::GetFusionNameByType(AnfAlgo::GetFusionType(cnode));
  (*compute_json)[kJModuleName] = kJModuleNamePrefix + func_name;
}

// node_out_idx: node output index
// desc_output_idx: this index use to add json
void TbeJsonCreator::GenDescJson(const AnfNodePtr &anf_node, size_t node_out_idx, size_t desc_output_idx,
                                 nlohmann::json *output_desc) {
  MS_EXCEPTION_IF_NULL(anf_node);
  GenDesJsonCommon(output_desc);
  std::vector<int64_t> shape;
  std::vector<int64_t> ori_shape;
  shape = TbeJsonUtils::GetOutputDeviceShapeForTbeBuild(anf_node, node_out_idx);
  ori_shape = TbeJsonUtils::GetOutputOriShapeForTbeBuild(anf_node, node_out_idx);
  if (shape.empty()) {
    shape.emplace_back(1);
  }
  if (ori_shape.empty()) {
    ori_shape.emplace_back(1);
  }

  auto full_name = anf_node->fullname_with_scope();
  auto output_desc_name = node_out_idx > 0 ? (full_name + "_" + std::to_string(node_out_idx)) : full_name;

  // !! Note: format: only data node's output use it
  auto format = AnfAlgo::GetOutputFormat(anf_node, node_out_idx);
  format = tbe::TbeAdapter::FormatPass(format, ori_shape.size());
  auto def_format = TbeJsonUtils::IsNeedChangeDefaultFormat(anf_node) ? kOpFormat_NCDHW : kOpFormat_NCHW;
  format =
    (def_format == kOpFormat_NCDHW && k3DFormatSet.find(format) == k3DFormatSet.end()) ? kOpFormat_NCDHW : format;

  (*output_desc)[kJDataType] = tbe::TypeIdToString(AnfAlgo::GetOutputDeviceDataType(anf_node, node_out_idx));
  (*output_desc)[kJDtype] = GetJsonValue<std::string>(*output_desc, kJDataType);
  (*output_desc)[kJFormat] = format;
  (*output_desc)[kJOriFormat] = def_format;
  (*output_desc)[kJOriShape] = ori_shape;
  (*output_desc)[kJShape] = shape;
  (*output_desc)[kJName] = output_desc_name;
  // !! Note: output_index, only node's output use it
  (*output_desc)[kJOutputIndex] = desc_output_idx;
}

void TbeJsonCreator::GenDesJsonCommon(nlohmann::json *output_desc) {
  MS_EXCEPTION_IF_NULL(output_desc);
  (*output_desc)[kJL1AddrOffset] = 0;
  (*output_desc)[kJL1FusionType] = -1;
  (*output_desc)[kJL1WorkspaceSize] = -1;
  (*output_desc)[kJAddrType] = 0;
  (*output_desc)[kJSliceOffset] = nlohmann::json::array();
  (*output_desc)[kJSplitIndex] = 0;
  (*output_desc)[kJTotalShape] = nlohmann::json::array();
  (*output_desc)[kJValidShape] = nlohmann::json::array();
}

void ParseConstValue(const mindspore::ValuePtr &value, nlohmann::json *json_obj) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    TypePtr data_type = tensor->Dtype();
    MS_EXCEPTION_IF_NULL(data_type);
    TypeId type_id = data_type->type_id();
    (*json_obj)[kJConstValueDtype] = tbe::TypeIdToString(type_id);
    switch (type_id) {
      case kNumberTypeInt8:
        (*json_obj)[kJConstValue] = TensorValueToVector<int8_t>(tensor);
        break;

      case kNumberTypeUInt8:
        (*json_obj)[kJConstValue] = TensorValueToVector<uint8_t>(tensor);
        break;

      case kNumberTypeInt16:
        (*json_obj)[kJConstValue] = TensorValueToVector<int16_t>(tensor);
        break;

      case kNumberTypeUInt16:
        (*json_obj)[kJConstValue] = TensorValueToVector<uint16_t>(tensor);
        break;

      case kNumberTypeInt32:
        (*json_obj)[kJConstValue] = TensorValueToVector<int32_t>(tensor);
        break;

      case kNumberTypeUInt32:
        (*json_obj)[kJConstValue] = TensorValueToVector<uint32_t>(tensor);
        break;

      case kNumberTypeInt64:
        (*json_obj)[kJConstValue] = TensorValueToVector<int64_t>(tensor);
        break;

      case kNumberTypeUInt64:
        (*json_obj)[kJConstValue] = TensorValueToVector<uint64_t>(tensor);
        break;

      case kNumberTypeFloat32:
        (*json_obj)[kJConstValue] = TensorValueToVector<float>(tensor);
        break;

      case kNumberTypeFloat64:
        (*json_obj)[kJConstValue] = TensorValueToVector<double>(tensor);
        break;

      default:
        MS_LOG(EXCEPTION) << "When parse const input value, the value data type: " << data_type << " is not supported.";
    }
  } else {
    MS_LOG(WARNING) << "Const value input is not a tensor.";
  }
}

void TbeJsonCreator::GenInputConstValue(const AnfNodePtr &anf_node, size_t real_input_index,
                                        nlohmann::json *input_desc) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_desc);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto value_depend = build_info->GetInputValueDepend(real_input_index);
  if (value_depend.empty() || value_depend == kIgnored) {
    return;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->inputs()[real_input_index + 1];
  if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimDepend)) {
    input_node = common::AnfAlgo::VisitKernel(input_node, 0).first;
  }
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<ValueNode>()) {
    MS_LOG(INFO) << "Const Input value node info : " << GetValueNode(input_node)->ToString();
    auto value_node = input_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    ParseConstValue(value, input_desc);
  } else if (input_node->isa<Parameter>()) {
    auto param = input_node->cast<ParameterPtr>();
    auto value = param->default_param();
    MS_EXCEPTION_IF_NULL(value);
    ParseConstValue(value, input_desc);
  } else {
    MS_LOG(ERROR) << "The operator " << anf_node->fullname_with_scope() << "'s input" << real_input_index
                  << "'s value depend is " << value_depend << ", but its input node is a " << input_node->type_name()
                  << ", not a value node.";
  }
}

void TbeJsonCreator::AttrsJsonPreProcessing(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *attrs_ptr,
                                            nlohmann::json *attrs_json) {
  tbe::TbeAdapter::CastAttrJsonPrePass(anf_node, attrs_ptr, attrs_json);
}
void TbeJsonCreator::GenOutputDataDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  auto op_desc = AnfAlgo::GetOutputDataDesc(anf_node);
  // get output_data_desc from prebuild
  if (!op_desc.empty() && op_desc.at(0).find(kJListArgs) != op_desc.at(0).end()) {
    (*compute_json)[kJOutputDataDesc] = GetJsonValue<nlohmann::json>(op_desc.at(0), kJListArgs);
  } else {
    auto outputs_desc = GetJsonValue<std::vector<nlohmann::json>>(*compute_json, kJOutputDesc);
    std::vector<nlohmann::json> outputs_data_desc;
    for (auto output_desc : outputs_desc) {
      if (output_desc.find(kJOriShape) != output_desc.end()) {
        output_desc.erase(kJName);
        outputs_data_desc.push_back(output_desc);
      }
    }
    (*compute_json)[kJOutputDataDesc] = outputs_data_desc;
  }
}

bool TbeJsonCreator::AttrsJsonPostProcessing(const AnfNodePtr &anf_node, const OpInfoPtr &op_info_ptr,
                                             nlohmann::json *attrs_json) {
  return true;
}
}  // namespace mindspore::kernel
