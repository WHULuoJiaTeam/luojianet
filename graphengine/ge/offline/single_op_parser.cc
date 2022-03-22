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
#include "offline/single_op_parser.h"

#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "framework/common/debug/ge_log.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_factory_impl.h"

using Json = nlohmann::json;
using std::string;
using std::vector;
using std::map;

namespace ge {
namespace {
constexpr char const *kKeyOp = "op";
constexpr char const *kKeyInputDesc = "input_desc";
constexpr char const *kKeyOutputDesc = "output_desc";
constexpr char const *kKeyAttr = "attr";
constexpr char const *kKeyName = "name";
constexpr char const *kKeyType = "type";
constexpr char const *kKeyShape = "shape";
constexpr char const *kKeyOriginShape = "origin_shape";
constexpr char const *kKeyShapeRange = "shape_range";
constexpr char const *kKeyValue = "value";
constexpr char const *kKeyFormat = "format";
constexpr char const *kKeyOriginFormat = "origin_format";
constexpr char const *kFileSuffix = ".om";
constexpr char const *kKeyDynamicInput = "dynamic_input";
constexpr char const *kKeyDynamicOutput = "dynamic_output";
constexpr char const *kKeyCompileFlag = "compile_flag";
constexpr int kDumpJsonIndent = 2;
constexpr int kShapeRangePairSize = 2;
constexpr int kShapeRangeLow = 0;
constexpr int kShapeRangeHigh = 1;
constexpr int kMaxFileNameLen = 128;

map<string, GeAttrValue::ValueType> kAttrTypeDict = {
    {"bool", GeAttrValue::VT_BOOL},
    {"int", GeAttrValue::VT_INT},
    {"float", GeAttrValue::VT_FLOAT},
    {"string", GeAttrValue::VT_STRING},
    {"list_bool", GeAttrValue::VT_LIST_BOOL},
    {"list_int", GeAttrValue::VT_LIST_INT},
    {"list_float", GeAttrValue::VT_LIST_FLOAT},
    {"list_string", GeAttrValue::VT_LIST_STRING},
    {"list_list_int", GeAttrValue::VT_LIST_LIST_INT},
    {"data_type", GeAttrValue::VT_DATA_TYPE},
};

map<string, DataType> kDataTypeDict = {
    {"bool", DT_BOOL},
    {"int8", DT_INT8},
    {"uint8", DT_UINT8},
    {"int16", DT_INT16},
    {"uint16", DT_UINT16},
    {"int32", DT_INT32},
    {"uint32", DT_UINT32},
    {"int64", DT_INT64},
    {"uint64", DT_UINT64},
    {"float16", DT_FLOAT16},
    {"half", DT_FLOAT16},
    {"fp16", DT_FLOAT16},
    {"float", DT_FLOAT},
    {"float32", DT_FLOAT},
    {"double", DT_DOUBLE},
    {"complex64", DT_COMPLEX64},
    {"complex128", DT_COMPLEX128}
};

map<string, Format> kFormatDict = {
    {"nchw", FORMAT_NCHW},
    {"nhwc", FORMAT_NHWC},
    {"nd", FORMAT_ND},
    {"nc1hwc0", FORMAT_NC1HWC0},
    {"fractal_z", FORMAT_FRACTAL_Z},
    {"nc1c0hwpad", FORMAT_NC1C0HWPAD},
    {"nhwc1c0", FORMAT_NHWC1C0},
    {"fsr_nchw", FORMAT_FSR_NCHW},
    {"fractal_deconv", FORMAT_FRACTAL_DECONV},
    {"c1hwnc0", FORMAT_C1HWNC0},
    {"fractal_deconv_transpose", FORMAT_FRACTAL_DECONV_TRANSPOSE},
    {"fractal_deconv_sp_stride_trans", FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
    {"nc1hwc0_c04", FORMAT_NC1HWC0_C04},
    {"fractal_z_c04", FORMAT_FRACTAL_Z_C04},
    {"chwn", FORMAT_CHWN},
    {"deconv_sp_stride8_trans", FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
    {"nc1khkwhwc0", FORMAT_NC1KHKWHWC0},
    {"bn_weight", FORMAT_BN_WEIGHT},
    {"filter_hwck", FORMAT_FILTER_HWCK},
    {"hwcn", FORMAT_HWCN},
    {"lookup_lookups", FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
    {"lookup_keys", FORMAT_HASHTABLE_LOOKUP_KEYS},
    {"lookup_value", FORMAT_HASHTABLE_LOOKUP_VALUE},
    {"lookup_output", FORMAT_HASHTABLE_LOOKUP_OUTPUT},
    {"lookup_hits", FORMAT_HASHTABLE_LOOKUP_HITS},
    {"md", FORMAT_MD},
    {"c1hwncoc0", FORMAT_C1HWNCoC0},
    {"fractal_nz", FORMAT_FRACTAL_NZ},
    {"ndhwc", FORMAT_NDHWC},
    {"ncdhw", FORMAT_NCDHW},
    {"dhwcn", FORMAT_DHWCN},
    {"dhwnc", FORMAT_DHWNC},
    {"ndc1hwc0", FORMAT_NDC1HWC0},
    {"fractal_z_3d", FORMAT_FRACTAL_Z_3D},
    {"fractal_z_3d_transpose", FORMAT_FRACTAL_Z_3D_TRANSPOSE},
    {"cn", FORMAT_CN},
    {"nc", FORMAT_NC},
    {"fractal_zn_lstm", FORMAT_FRACTAL_ZN_LSTM},
    {"fractal_z_g", FORMAT_FRACTAL_Z_G}
};

std::string GenerateFileName(const SingleOpDesc &single_op_desc, int index) {
  std::stringstream file_name_ss;
  file_name_ss << index;
  file_name_ss << "_" << single_op_desc.op;
  for (auto &desc : single_op_desc.input_desc) {
    file_name_ss << "_" << desc.type << "_" << desc.format;
    for (auto dim : desc.dims) {
      file_name_ss << "_" << dim;
    }
  }

  for (auto &desc : single_op_desc.output_desc) {
    file_name_ss << "_" << desc.type << "_" << desc.format;
    for (auto dim : desc.dims) {
      file_name_ss << "_" << dim;
    }
  }

  std::string file_name = file_name_ss.str();
  if (file_name.length() > kMaxFileNameLen) {
    GELOGI("Trim file name for it is too long, origin file name = %s", file_name.c_str());
    file_name = file_name.substr(0, kMaxFileNameLen);
  }
  file_name += kFileSuffix;
  return file_name;
}
}  // namespace

bool AttrValueIsString(const Json &j, const string &key) {
  try {
    string tmp_str = j.at(key).get<string>();
    return true;
  } catch (Json::type_error &e) {
    return false;
  }
}

template<typename T>
T GetValue(const map<string, T> &dict, string &key, T default_val) {
  transform(key.begin(), key.end(), key.begin(), ::tolower);
  auto it = dict.find(key);
  if (it == dict.end()) {
    return default_val;
  }

  return it->second;
}

template<typename T>
void SetAttrValue(const Json &j, SingleOpAttr &attr) {
  // when attr type is "data_type", we support two kinds of attr value.
  // 1. value: "DT_FLOAT", "DT_INT32", "DT_INT8" ...
  // 2. value: 1, 3 ...
  if (j.at(kKeyType).get<string>() == "data_type" && AttrValueIsString(j, kKeyValue)) {
    string type_str = j.at(kKeyValue).get<string>();
    DataType dtype = TypeUtils::SerialStringToDataType(type_str);
    attr.value.SetValue<DataType>(dtype);
    return;
  }
  attr.value.SetValue<T>(j.at(kKeyValue).get<T>());
}

void from_json(const Json &j, SingleOpTensorDesc &desc) {
  bool is_tensor_valid = true;
  desc.dims = j.at(kKeyShape).get<vector<int64_t>>();
  auto it = j.find(kKeyShapeRange);
  if (it != j.end()) {
    desc.dim_ranges = j.at(kKeyShapeRange).get<vector<std::vector<int64_t>>>();
  }
  it = j.find(kKeyOriginShape);
  if (it != j.end()) {
    desc.ori_dims = j.at(kKeyOriginShape).get<vector<int64_t>>();
  }
  string format_str = j.at(kKeyFormat).get<string>();
  string type_str = j.at(kKeyType).get<string>();
  desc.format = GetValue(kFormatDict, format_str, FORMAT_RESERVED);
  desc.type = GetValue(kDataTypeDict, type_str, DT_UNDEFINED);
  is_tensor_valid = is_tensor_valid && ge::TypeUtils::IsFormatValid(format_str);
  is_tensor_valid = is_tensor_valid && ge::TypeUtils::IsDataTypeValid(type_str);
  it = j.find(kKeyOriginFormat);
  if (it != j.end()) {
    string origin_format_str = j.at(kKeyOriginFormat).get<string>();
    is_tensor_valid = is_tensor_valid && ge::TypeUtils::IsFormatValid(origin_format_str);
    desc.ori_format = GetValue(kFormatDict, origin_format_str, FORMAT_RESERVED);
  }
  auto tensor_name = j.find(kKeyName);
  if (tensor_name != j.end()) {
    desc.name = tensor_name->get<string>();
  }
  auto dynamic_input_name = j.find(kKeyDynamicInput);
  if (dynamic_input_name != j.end()) {
    desc.dynamic_input_name = dynamic_input_name->get<string>();
  }
  if (!is_tensor_valid) {
    desc.SetValidFlag(is_tensor_valid);
  }
}

void from_json(const Json &j, SingleOpAttr &attr) {
  attr.name = j.at(kKeyName).get<string>();
  attr.type = j.at(kKeyType).get<string>();
  auto it = kAttrTypeDict.find(attr.type);
  if (it == kAttrTypeDict.end()) {
    GELOGE(UNSUPPORTED, "[Find][JsonAttr] name=%s, type=%s failed for Unsupported type.",
        attr.name.c_str(), attr.type.c_str());
    REPORT_INNER_ERROR("E19999", "Find jsonattr name=%s, type=%s failed for Unsupported type.",
        attr.name.c_str(), attr.type.c_str());
    return;
  }

  switch (it->second) {
    case GeAttrValue::VT_BOOL:
      SetAttrValue<bool>(j, attr);
      break;
    case GeAttrValue::VT_INT:
      SetAttrValue<int64_t>(j, attr);
      break;
    case GeAttrValue::VT_FLOAT:
      SetAttrValue<float>(j, attr);
      break;
    case GeAttrValue::VT_STRING:
      SetAttrValue<string>(j, attr);
      break;
    case GeAttrValue::VT_LIST_BOOL:
      SetAttrValue<vector<bool>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_INT:
      SetAttrValue<vector<int64_t>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_FLOAT:
      SetAttrValue<vector<float>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_STRING:
      SetAttrValue<vector<string>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_LIST_INT:
      SetAttrValue<vector<vector<int64_t>>>(j, attr);
      break;
    case GeAttrValue::VT_DATA_TYPE:
      SetAttrValue<DataType>(j, attr);
      break;
    default:
      GELOGE(UNSUPPORTED, "[Find][JsonAttr] name=%s, type=%s failed for Unsupported type.",
          attr.name.c_str(), attr.type.c_str());
      REPORT_INNER_ERROR("E19999", "Find jsonattr name=%s, type=%s failed for Unsupported type.",
          attr.name.c_str(), attr.type.c_str());
      break;
  }
}

void from_json(const Json &j, SingleOpDesc &desc) {
  auto op = j.find(kKeyOp);
  if (op != j.end()) {
    desc.op = j.at(kKeyOp).get<string>();
  }

  auto input_desc = j.find(kKeyInputDesc);
  if (input_desc != j.end()) {
    desc.input_desc = input_desc->get<vector<SingleOpTensorDesc>>();
  }

  auto output_desc = j.find(kKeyOutputDesc);
  if (output_desc != j.end()) {
    desc.output_desc = output_desc->get<vector<SingleOpTensorDesc>>();
  }

  auto attr_field = j.find(kKeyAttr);
  if (attr_field != j.end()) {
    desc.attrs = attr_field->get<vector<SingleOpAttr>>();
  }

  auto compile_flag = j.find(kKeyCompileFlag);
  if (compile_flag != j.end()) {
    desc.compile_flag = compile_flag->get<int32_t>();
  }
}

Status SingleOpParser::ReadJsonFile(const std::string &file, Json &json_obj) {
  std::string real_path = RealPath(file.c_str());
  if (real_path.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10023", {"value"}, {file});
    GELOGE(FAILED, "[Read][JsonFile]Input parameter[--singleop]'s value[%s] is not a valid path.", file.c_str());
    return INTERNAL_ERROR;
  }

  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10024", {"value"}, {file});
    GELOGE(FAILED, "[Open][JsonFile] failed for file[%s] provided in input parameter[--singleop].", file.c_str());
    return FAILED;
  }
  try {
    ifs >> json_obj;
  } catch (const std::exception &e) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10025", {"realpath", "errmsg"}, {real_path, e.what()});
    GELOGE(PARAM_INVALID,
        "[Parse][JsonFile] fail for file[%s] provided in input parameter[--singleop], exception = %s.",
        real_path.c_str(), e.what());
    return PARAM_INVALID;
  }

  ifs.close();
  return SUCCESS;
}

bool SingleOpParser::Validate(const SingleOpDesc &op_desc) {
  if (op_desc.op.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10026");
    GELOGE(PARAM_INVALID, "[Check][Param] fail for name of input SingleOpDesc is empty.");
    return false;
  }

  int index = 0;
  for (auto &tensor_desc : op_desc.input_desc) {
    if (!tensor_desc.GetValidFlag()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10027", {"op_name", "input", "type", "index"},
          {op_desc.op, "input", "tensor", std::to_string(index)});
      GELOGE(PARAM_INVALID,
          "[Check][Param] fail for Input's dataType or format is invalid when the index is %d", index);
      return false;
    }
    if ((tensor_desc.type == DT_UNDEFINED && tensor_desc.format != FORMAT_RESERVED) ||
        (tensor_desc.type != DT_UNDEFINED && tensor_desc.format == FORMAT_RESERVED)){
      ErrorManager::GetInstance().ATCReportErrMessage("E10027", {"op_name", "input", "type", "index"},
          {op_desc.op, "input", "datatype or format", std::to_string(index)});
      GELOGE(PARAM_INVALID, "[Check][Param]Input's dataType or format is invalid when the index is %d", index);
      return false;
    }
    ++index;
  }

  index = 0;
  for (auto &tensor_desc : op_desc.output_desc) {
    if (!tensor_desc.GetValidFlag()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10027", {"op_name", "input", "type", "index"},
          {op_desc.op, "output", "tensor", std::to_string(index)});
      GELOGE(PARAM_INVALID, "[Check][Param]fail for Output's dataType is invalid when the index is %d", index);
      return false;
    }
    if (tensor_desc.type == DT_UNDEFINED) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10027", {"op_name", "input", "type", "index"},
          {op_desc.op, "output", "datatype", std::to_string(index)});
      GELOGE(PARAM_INVALID, "[Check][Param]Output's dataType is invalid when the index is %d", index);
      return false;
    }

    if (tensor_desc.format == FORMAT_RESERVED) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10027", {"op_name", "input", "type", "index"},
          {op_desc.op, "output", "format", std::to_string(index)});
      GELOGE(PARAM_INVALID, "[Check][Param]Output's format is invalid when the index is %d", index);
      return false;
    }
    ++index;
  }

  for (auto &attr : op_desc.attrs) {
    if (attr.name.empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10029", {"op_name"}, {op_desc.op});
      GELOGE(PARAM_INVALID, "[Parse][Attr]attr name is empty");
      return false;
    }

    if (attr.value.IsEmpty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10030", {"op_name", "attrname"}, {op_desc.op, attr.name});
      GELOGE(PARAM_INVALID, "[Parse][Attr] fail for vale of attr name:\"%s\" is empty. ", attr.name.c_str());
      return false;
    }
  }

  return true;
}

std::unique_ptr<OpDesc> SingleOpParser::CreateOpDesc(const string &op_type) {
  return std::unique_ptr<OpDesc>(new(std::nothrow) OpDesc(op_type, op_type));
}

Status SingleOpParser::UpdateDynamicTensorName(std::vector<SingleOpTensorDesc> &desc) {
  std::map<std::string, int> dynamic_name_map;
  for (auto &tensor : desc) {
    if (tensor.dynamic_input_name.empty()) {
      continue;
    }
    if (dynamic_name_map.find(tensor.dynamic_input_name) == dynamic_name_map.end()) {
      dynamic_name_map[tensor.dynamic_input_name] = 0;
    } else {
      dynamic_name_map[tensor.dynamic_input_name]++;
    }
    tensor.name = tensor.dynamic_input_name + std::to_string(dynamic_name_map[tensor.dynamic_input_name]);
  }
  GELOGD("Update dynamic tensor name success!");
  return SUCCESS;
}

Status SingleOpParser::ConvertToBuildParam(int index,
                                           const SingleOpDesc &single_op_desc,
                                           SingleOpBuildParam &build_param) {
  auto op_desc = CreateOpDesc(single_op_desc.op);
  GE_CHECK_NOTNULL(op_desc);

  for (auto &desc : single_op_desc.input_desc) {
    GeTensorDesc ge_tensor_desc(GeShape(desc.dims),
                                desc.format,
                                desc.type);
    auto ori_format_to_set = desc.ori_format != FORMAT_RESERVED ? desc.ori_format : desc.format;
    auto ori_dims = !desc.ori_dims.empty() ? desc.ori_dims : desc.dims;
    ge_tensor_desc.SetOriginFormat(ori_format_to_set);
    ge_tensor_desc.SetOriginShape(GeShape(ori_dims));
    GE_CHK_STATUS_RET_NOLOG(SetShapeRange(op_desc->GetName(), desc, ge_tensor_desc));
    TensorUtils::SetRealDimCnt(ge_tensor_desc, ori_dims.size());
    TensorUtils::SetInputTensor(ge_tensor_desc, true);
    TensorUtils::SetOutputTensor(ge_tensor_desc, false);
    if (desc.name.empty()) {
      op_desc->AddInputDesc(ge_tensor_desc);
    } else {
      op_desc->AddInputDesc(desc.name, ge_tensor_desc);
    }
    build_param.inputs.emplace_back(ge_tensor_desc);
  }

  for (auto &desc : single_op_desc.output_desc) {
    GeTensorDesc ge_tensor_desc(GeShape(desc.dims),
                                desc.format,
                                desc.type);
    auto ori_format_to_set = desc.ori_format != FORMAT_RESERVED ? desc.ori_format : desc.format;
    auto ori_dims = !desc.ori_dims.empty() ? desc.ori_dims : desc.dims;
    ge_tensor_desc.SetOriginFormat(ori_format_to_set);
    ge_tensor_desc.SetOriginShape(GeShape(ori_dims));
    GE_CHK_STATUS_RET_NOLOG(SetShapeRange(op_desc->GetName(), desc, ge_tensor_desc));
    TensorUtils::SetRealDimCnt(ge_tensor_desc, ori_dims.size());
    TensorUtils::SetInputTensor(ge_tensor_desc, false);
    TensorUtils::SetOutputTensor(ge_tensor_desc, true);
    if (desc.name.empty()) {
      op_desc->AddOutputDesc(ge_tensor_desc);
    } else {
      op_desc->AddOutputDesc(desc.name, ge_tensor_desc);
    }
    build_param.outputs.emplace_back(ge_tensor_desc);
  }

  for (const auto &attr : single_op_desc.attrs) {
    op_desc->SetAttr(attr.name, attr.value);
  }

  if (VerifyOpInputOutputSizeByIr(*op_desc) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Verify][OpInputOutputSize] fail for input op [%s] invalid.", op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  build_param.file_name = GenerateFileName(single_op_desc, index);
  build_param.op_desc.reset(op_desc.release());
  return SUCCESS;
}

Status SingleOpParser::VerifyOpInputOutputSizeByIr(const OpDesc &current_op_desc) {
  ge::Operator operator_ir = ge::OperatorFactory::CreateOperator("tmp_operator", current_op_desc.GetType());
  if (!operator_ir.IsEmpty()) {
    auto opdesc_ir = ge::OpDescUtils::GetOpDescFromOperator(operator_ir);
    GE_CHECK_NOTNULL(opdesc_ir);
    size_t current_opdesc_inputs_num = current_op_desc.GetInputsSize();
    size_t ir_opdesc_inputs_num = opdesc_ir->GetInputsSize();
    if (current_opdesc_inputs_num < ir_opdesc_inputs_num) {
      string reason = "is smaller than the ir needed input size " + std::to_string(ir_opdesc_inputs_num);
      ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
          {current_op_desc.GetName(), "input size " + std::to_string(current_opdesc_inputs_num), reason});
      GELOGE(PARAM_INVALID,
          "[Verify][OpInputOutputSize]This op:%s input size %zu is smaller than the ir needed input size %zu",
          current_op_desc.GetName().c_str(), current_opdesc_inputs_num, ir_opdesc_inputs_num);
      return PARAM_INVALID;
    }
    size_t current_opdesc_outputs_num = current_op_desc.GetOutputsSize();
    size_t ir_opdesc_outputs_num = opdesc_ir->GetOutputsSize();
    if (current_opdesc_outputs_num < ir_opdesc_outputs_num) {
      string reason = "is smaller than the ir needed output size " + std::to_string(ir_opdesc_outputs_num);
      ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
          {current_op_desc.GetName(), "output size " + std::to_string(current_opdesc_outputs_num), reason});
      GELOGE(PARAM_INVALID,
          "[Verify][OpInputOutputSize]This op:%s output size %zu is smaller than the ir needed output size %zu",
          current_op_desc.GetName().c_str(), current_opdesc_outputs_num, ir_opdesc_outputs_num);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status SingleOpParser::SetShapeRange(const std::string &op_name,
                                     const SingleOpTensorDesc &tensor_desc,
                                     GeTensorDesc &ge_tensor_desc) {
  auto num_shape_ranges = tensor_desc.dim_ranges.size();
  GELOGD("Number of shape ranges = %zu", num_shape_ranges);
  auto it = std::find(tensor_desc.dims.begin(), tensor_desc.dims.end(), ge::UNKNOWN_DIM_NUM);
  if (it != tensor_desc.dims.end()) {
    if (tensor_desc.dims != ge::UNKNOWN_RANK) {
      ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
                                                      {op_name,
                                                       "shape",
                                                       "has unknown rank but dim size is not one"});
      GELOGE(PARAM_INVALID, "[Set][ShapeRange]Invalid tensor shape:%s.",
          ge_tensor_desc.MutableShape().ToString().c_str());
      return PARAM_INVALID;
    }
    if (!tensor_desc.dim_ranges.empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
                                                      {op_name,
                                                       "shape range",
                                                       "is not needed while the rank the shape is unknown"});
      GELOGE(PARAM_INVALID, "[Set][ShapeRange]Shape range is not needed while the rank the shape is unknown.");
      return PARAM_INVALID;
    }

    GELOGD("Shape is unknown rank, do not set shape range");
    return SUCCESS;
  }

  std::vector<std::pair<int64_t, int64_t>> shape_range;
  size_t range_index = 0;
  for (auto dim : tensor_desc.dims) {
    if (dim >= 0) {
      shape_range.emplace_back(dim, dim);
      GELOGD("Adding shape range: [%ld, %ld]", dim, dim);
    } else {
      GELOGD("To get shape range by index = %zu", range_index);
      if (range_index >= num_shape_ranges) {
        string reason = "is smaller than the unknown dim size " + std::to_string(++range_index);
        ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
                                                        {op_name,
                                                         "shape range size " + std::to_string(num_shape_ranges),
                                                         reason});
        GELOGE(PARAM_INVALID, "[Set][ShapeRange]The number of shape_range mismatches that of unknown dims.");
        return PARAM_INVALID;
      }

      auto &range = tensor_desc.dim_ranges[range_index];
      if (range.size() != kShapeRangePairSize) {
        string reason = "has " + std::to_string(range.size()) + " item(s)";
        ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
                                                        {op_name,
                                                         "shape range " + std::to_string(range_index),
                                                         reason});
        GELOGE(PARAM_INVALID, "[Set][ShapeRange]Invalid shape range entry. index = %zu, size = %zu",
            range_index, range.size());
        return PARAM_INVALID;
      }

      shape_range.emplace_back(range[kShapeRangeLow], range[kShapeRangeHigh]);
      GELOGD("Adding shape range: [%ld, %ld]", range[kShapeRangeLow], range[kShapeRangeHigh]);
      ++range_index;
    }
  }

  if (num_shape_ranges != range_index) {
    string reason = "is greater than the unknown dim size " + std::to_string(range_index);
    ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
                                                    {op_name,
                                                     "shape range size " + std::to_string(num_shape_ranges),
                                                     reason});
    GELOGE(PARAM_INVALID,
        "[Set][ShapeRange]The number of shape_range(%zu) mismatches that of unknown dims(%zu).",
        num_shape_ranges, range_index);
    return PARAM_INVALID;
  }

  if (range_index > 0) {
    ge_tensor_desc.SetShapeRange(shape_range);
  }

  return SUCCESS;
}

Status SingleOpParser::ParseSingleOpList(const std::string &file, std::vector<SingleOpBuildParam> &op_list) {
  int index = 0;
  try {
    Json single_op_list_json;
    auto ret = ReadJsonFile(file, single_op_list_json);
    if (ret != SUCCESS) {
      return ret;
    }

    int32_t compile_flag = 0;
    for (const Json &single_op_json : single_op_list_json) {
      SingleOpDesc single_op_desc;
      GELOGI("Parsing op[%d], jsonStr = %s", index, single_op_json.dump(kDumpJsonIndent).c_str());
      single_op_desc = single_op_json;
      GELOGD("Compile flag is %d.", single_op_desc.compile_flag);
      if (single_op_desc.compile_flag == 1) {
        compile_flag = single_op_desc.compile_flag;
        continue;
      }
      if (UpdateDynamicTensorName(single_op_desc.input_desc) != SUCCESS) {
        GELOGE(FAILED, "[Update][DynamicTensorName] failed for invalid input param!");
        REPORT_CALL_ERROR("E19999", "UpdateDynamicTensorName failed for invalid input param.");
        return FAILED;
      }

      if (!Validate(single_op_desc)) {
        GELOGE(PARAM_INVALID,
            "[Check][OpDesc]Validate the index[%d] of op failed when read json file[%s].", index, file.c_str());
        return PARAM_INVALID;
      }

      SingleOpBuildParam param;
      ret = ConvertToBuildParam(index, single_op_desc, param);
      if (ret != SUCCESS) {
        return ret;
      }
      param.compile_flag = compile_flag;

      op_list.emplace_back(param);
      GELOGI("Parse the index[%d] of op success", index);
      index += 1;
    }
  } catch (const nlohmann::json::exception &e) {
    REPORT_INNER_ERROR("E19999", "parse singleop file:%s failed, catch exception:%s, current index:%d",
                       file.c_str(), e.what(), index);
    GELOGE(PARAM_INVALID, "[Parse][OpList] the index:%d of op failed when read json file:%s, exception:%s",
        index, file.c_str(), e.what());
    return PARAM_INVALID;
  }

  return SUCCESS;
}
} // namespace ge

