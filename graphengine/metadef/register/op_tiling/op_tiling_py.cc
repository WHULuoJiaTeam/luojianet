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

#include <nlohmann/json.hpp>
#include "securec.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_log.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"

namespace optiling {
thread_local int64_t last_op_tiling_perf = -1;

void ParseShapeDesc(const nlohmann::json &shape, std::vector<TeOpTensor> &tensors) {
  TeOpTensor tensor;
  if (shape.contains("shape")) {
    tensor.shape = shape["shape"].get<std::vector< int64_t>>();
  }
  if (shape.contains("ori_shape")) {
    tensor.ori_shape = shape["ori_shape"].get<std::vector<int64_t>>();
  }
  if (shape.contains("format")) {
    tensor.format = shape["format"].get<std::string>();
  }
  if (shape.contains("ori_format")) {
    tensor.ori_format = shape["ori_format"].get<std::string>();
  }
  if (shape.contains("dtype")) {
    tensor.dtype = shape["dtype"].get<std::string>();
  }
  tensors.emplace_back(tensor);
}

void ParseShapeDescList(const nlohmann::json &shape_list, std::vector<TeOpTensorArg> &op_args) {
  for (const auto &elem : shape_list) {
    TeOpTensorArg tensor_arg;
    tensor_arg.arg_type = TA_NONE;

    if (elem.is_array()) {
      tensor_arg.arg_type = TA_LIST;
      for (const auto &shape : elem) {
        ParseShapeDesc(shape, tensor_arg.tensor);
      }
    } else {
      tensor_arg.arg_type = TA_SINGLE;
      ParseShapeDesc(elem, tensor_arg.tensor);
    }
    op_args.emplace_back(tensor_arg);
  }
}

void ParseShapeDescV2(const nlohmann::json &shape, ge::OpDescPtr &op_desc, const bool &is_input) {
  ge::GeTensorDesc tensor;
  std::string name;
  if (shape.contains("shape")) {
    tensor.SetShape(ge::GeShape(shape["shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("ori_shape")) {
    tensor.SetOriginShape(ge::GeShape(shape["ori_shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("format")) {
    std::string format_str = shape["format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetFormat(ge_format);
  }
  if (shape.contains("ori_format")) {
    std::string format_str = shape["ori_format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetOriginFormat(ge_format);
  }
  if (shape.contains("dtype")) {
    std::string dtype_str = shape["dtype"].get<std::string>();
    std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
    tensor.SetDataType(ge_dtype);
  }
  if (shape.contains("name")) {
    name = shape["name"];
    tensor.SetName(name);
    is_input ? op_desc->AddInputDesc(name, tensor) : op_desc->AddOutputDesc(name, tensor);
  } else {
    is_input ? op_desc->AddInputDesc(tensor) : op_desc->AddOutputDesc(tensor);
  }
}

void ParseShapeDescListV2(const nlohmann::json &shape_list, ge::OpDescPtr &op_desc, const bool &is_input) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseShapeDescV2(shape, op_desc, is_input);
      }
    } else {
      ParseShapeDescV2(elem, op_desc, is_input);
    }
  }
}

template<typename T>
void GetConstDataPointer(const nlohmann::json &json_array, std::vector<uint8_t> &const_value) {
  std::vector<T> value = json_array.get<std::vector<T>>();
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(value.data());
  uint8_t *pv_end = pv_begin + value.size() * sizeof(T);
  const_value = std::vector<uint8_t>(pv_begin, pv_end);
}

void CopyConstDataWithFloat16(const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  std::vector<float> const_value = json_array.get<std::vector<float>>();
  float *const_data_ptr = const_value.data();
  if (const_data_ptr == nullptr) {
    GE_LOGE("Get const data pointer failed");
    return;
  }
  std::vector<uint16_t> const_data_vec;
  size_t size = sizeof(const_value)/sizeof(float);
  for (size_t i = 0; i < size; ++i) {
    float const_data = *(const_data_ptr + i);
    uint16_t const_data_uint16 = FloatToUint16(const_data);
    const_data_vec.emplace_back(const_data_uint16);
  }
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(const_data_vec.data());
  uint8_t *pv_end = pv_begin + const_data_vec.size() * sizeof(uint16_t);
  value = std::vector<uint8_t>(pv_begin, pv_end);
}

bool CopyConstData(const std::string &dtype, const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  if (dtype == "int8") {
    GetConstDataPointer<int8_t>(json_array, value);
  } else if (dtype == "uint8") {
    GetConstDataPointer<uint8_t>(json_array, value);
  } else if (dtype == "int16") {
    GetConstDataPointer<int16_t>(json_array, value);
  } else if (dtype == "uint16") {
    GetConstDataPointer<uint16_t>(json_array, value);
  } else if (dtype == "int32") {
    GetConstDataPointer<int32_t>(json_array, value);
  } else if (dtype == "uint32") {
    GetConstDataPointer<uint32_t>(json_array, value);
  } else if (dtype == "int64") {
    GetConstDataPointer<int64_t>(json_array, value);
  } else if (dtype == "uint64") {
    GetConstDataPointer<uint64_t>(json_array, value);
  } else if (dtype == "float32") {
    GetConstDataPointer<float>(json_array, value);
  } else if (dtype == "double") {
    GetConstDataPointer<double>(json_array, value);
  } else if (dtype == "float16") {
    CopyConstDataWithFloat16(json_array, value);
  } else {
    GE_LOGE("Unknown dtype: %s", dtype.c_str());
    return false;
  }
  return true;
}

void ParseConstShapeDesc(const nlohmann::json &shape_json, std::map<std::string, TeConstTensorData> &const_tensors,
                         std::map<std::string, std::vector<uint8_t>> &const_values) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not const tenosr");
    return;
  }
  if (!shape_json.contains("name")) {
    GE_LOGE("const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<std::vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    GE_LOGE("CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  ge::Shape ge_shape(shape);
  std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
  dtype_str = "DT_" + dtype_str;
  ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
  ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge_format, ge_dtype), res.first->second);
  const_tensors.emplace(name, std::make_tuple(const_tensor.GetData(), const_tensor.GetSize(), const_tensor));
  return;
}

void ParseConstTensorList(const nlohmann::json &shape_list, std::map<std::string, TeConstTensorData> &const_tensors,
                          std::map<std::string, std::vector<uint8_t>> &const_values) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDesc(shape, const_tensors, const_values);
      }
    } else {
      ParseConstShapeDesc(elem, const_tensors, const_values);
    }
  }
}

void ParseConstShapeDescV2(const nlohmann::json &shape_json, ge::Operator &op_para,
                           std::map<std::string, std::vector<uint8_t>> &const_values, ge::OpDescPtr op_desc) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not const tenosr");
    return;
  }
  if (!shape_json.contains("name")) {
    REPORT_CALL_ERROR("E19999", "const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<std::vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    REPORT_CALL_ERROR("E19999", "CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  ge::GeShape ge_shape(shape);
  std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
  dtype_str = "DT_" + dtype_str;
  ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
  ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  ge::GeTensorDesc ge_tensor(ge_shape, ge_format, ge_dtype);
  ge_tensor.SetName(name);
  ge::GeTensor const_tensor(ge_tensor, res.first->second);
  ge::GeTensorPtr const_tensor_ptr = std::make_shared<ge::GeTensor>(const_tensor);
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_tensor_ptr);
  ge::Operator const_op = ge::OpDescUtils::CreateOperatorFromOpDesc(const_op_desc);
  op_para.SetInput(name, const_op);
  return;
}

void ParseConstTensorListV2(const nlohmann::json &shape_list, ge::Operator &operator_para,
                            std::map<std::string, std::vector<uint8_t>> &const_values, ge::OpDescPtr op_desc) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDescV2(shape, operator_para, const_values, op_desc);
      }
    } else {
      ParseConstShapeDescV2(elem, operator_para, const_values, op_desc);
    }
  }
}

std::string DumpByteBuffer(const ByteBuffer &buf) {
  static const char hex_digits[] = "0123456789ABCDEF";
  std::string str = buf.str();
  std::string output;
  output.reserve(str.size() * 2);
  for (unsigned char c : str) {
    output.push_back(hex_digits[c >> 4]);
    output.push_back(hex_digits[c & 15]);
  }
  return output;
}

bool DumpRunInfo(const OpRunInfo &run_info, char *run_info_json, const size_t &run_info_len) {
  if (run_info_json == nullptr) {
    GE_LOGE("run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  json_obj["block_dim"] = run_info.block_dim;
  json_obj["workspaces"] = run_info.workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.tiling_data);
  json_obj["clear_atomic"] = run_info.clear_atomic;
  json_obj["tiling_key"] = run_info.tiling_key;

  std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    GE_LOGE("runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  return memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1) == EOK;
}

bool DumpRunInfoV2(const OpRunInfoV2 &run_info, char *run_info_json, const size_t &run_info_len) {
  if (run_info_json == nullptr) {
    REPORT_CALL_ERROR("E19999", "run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  std::vector<int64_t> workspaces;
  int64_t workspace;
  for (size_t i = 0; i < run_info.GetWorkspaceNum(); ++i) {
    (void) run_info.GetWorkspace(i, workspace);
    workspaces.push_back(workspace);
  }
  json_obj["block_dim"] = run_info.GetBlockDim();
  json_obj["workspaces"] = workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.GetAllTilingData());
  json_obj["clear_atomic"] = run_info.GetClearAtomic();
  json_obj["tiling_key"] = run_info.GetTilingKey();

  std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    REPORT_CALL_ERROR("E19999", "runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  return memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1) == EOK;
}

extern "C" int TbeOpTilingPyInterfaceEx2BackUp(const char *optype, const char *compile_info, const char *inputs,
                                               const char *outputs, char *run_info_json, size_t run_info_len,
                                               const char *compile_info_hash, uint64_t *elapse,
                                               const OpTilingFunc &tiling_func) {
  if (optype == nullptr || compile_info == nullptr || inputs == nullptr || outputs == nullptr) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }

  std::chrono::time_point<std::chrono::steady_clock> before_tiling, after_tiling;
  std::string compile_info_str = compile_info;
  TeOpParas op_params;
  op_params.op_type = optype;
  std::map<std::string, std::vector<uint8_t>> const_values;
  try {
    nlohmann::json inputs_json = nlohmann::json::parse(inputs);
    nlohmann::json outputs_json = nlohmann::json::parse(outputs);
    ParseShapeDescList(inputs_json, op_params.inputs);
    ParseShapeDescList(outputs_json, op_params.outputs);
    ParseConstTensorList(inputs_json, op_params.const_inputs, const_values);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func found, op_type:%s", optype);

  OpCompileInfo op_compile_info{compile_info};
  if (compile_info_hash) {
    op_compile_info.key = compile_info_hash;
  }

  OpRunInfo run_info;
  if (elapse) {
    before_tiling = std::chrono::steady_clock::now();
  }

  bool rc = (tiling_func)(op_params, op_compile_info, run_info);

  if (elapse) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse) {
    *elapse = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();
    *(elapse + 1) = last_op_tiling_perf;
    last_op_tiling_perf = -1;
  }

  GELOGI("Optiling succeed. op_type:%s", optype);
  DumpRunInfo(run_info, run_info_json, run_info_len);
  return 1;
}

extern "C" int TbeOpTilingPyInterfaceEx2New(const char *optype, const char *compile_info, const char *inputs,
                                            const char *outputs, char *run_info_json, size_t run_info_len,
                                            const char *compile_info_hash, uint64_t *elapse,
                                            const OpTilingFuncV2 &tiling_func) {
  if (optype == nullptr || compile_info == nullptr || inputs == nullptr || outputs == nullptr) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }

  GELOGI("Optiling func v2 found, op_type:%s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling, after_tiling;
  std::string compile_info_str = compile_info;
  std::string optype_str = optype;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    nlohmann::json inputs_json = nlohmann::json::parse(inputs);
    nlohmann::json outputs_json = nlohmann::json::parse(outputs);
    ParseShapeDescListV2(inputs_json, op_desc, true);
    ParseShapeDescListV2(outputs_json, op_desc, false);
    operator_param = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    ParseConstTensorListV2(inputs_json, operator_param, const_values, op_desc);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }

  OpCompileInfoV2 op_compile_info{" ", compile_info_str};
  ge::AscendString opCompileInfoHash(compile_info_hash);
  if (compile_info_hash) {
    op_compile_info.SetKey(opCompileInfoHash);
  }

  OpRunInfoV2 run_info(uint32_t(0), false, uint64_t(0));
  if (elapse) {
    before_tiling = std::chrono::steady_clock::now();
  }

  bool rc = (tiling_func)(operator_param, op_compile_info, run_info);

  if (elapse) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse) {
    *elapse = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();
    *(elapse + 1) = last_op_tiling_perf;
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v2 succeed. op_type:%s", optype);
  DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

extern "C" int TbeOpTilingPyInterfaceEx3(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse,
                                         const OpTilingFuncV3 &tiling_func, const OpParseFuncV3 &parse_func) {
  if (optype == nullptr || compile_info == nullptr || inputs == nullptr || outputs == nullptr) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }

  GELOGI("Optiling func v3 found, op_type:%s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling, after_tiling;
  std::string compile_info_str = compile_info;
  std::string optype_str = optype;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    nlohmann::json inputs_json = nlohmann::json::parse(inputs);
    nlohmann::json outputs_json = nlohmann::json::parse(outputs);
    ParseShapeDescListV2(inputs_json, op_desc, true);
    ParseShapeDescListV2(outputs_json, op_desc, false);
    operator_param = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    ParseConstTensorListV2(inputs_json, operator_param, const_values, op_desc);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  if (compile_info_hash == nullptr) {
    return 0;
  }

  ge::AscendString compile_info_json_str = compile_info;
  void* op_compile_json_ptr = (parse_func)(operator_param, compile_info_json_str);

  OpRunInfoV2 run_info(uint32_t(0), false, uint64_t(0));
  if (elapse) {
    before_tiling = std::chrono::steady_clock::now();
  }
  bool rc = (tiling_func)(operator_param, op_compile_json_ptr, run_info);

  if (elapse) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse) {
    *elapse = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();
    *(elapse + 1) = last_op_tiling_perf;
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v3 succeed. op_type:%s", optype);
  DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

extern "C" int TbeOpTilingPyInterfaceEx2(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse) {
  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find(optype);
  if (iter == op_func_map.end()) {
    GELOGI("Op tiling function is not found by op type[%s].", optype);
    iter = op_func_map.find(OP_TYPE_AUTO_TILING);
    if (iter == op_func_map.end()) {
      GELOGI("Optiling func of op type[%s] is not found by Autotiling.", optype);
      REPORT_CALL_ERROR("E19999", "Optiling func is not found. op_type:%s", optype);
      return ge::GRAPH_FAILED;
    }
  }
  OpTilingFuncInfo &op_func_info = iter->second;
  int ret = 0;
  if (op_func_info.IsFunctionV3()) {
    const OpTilingFuncV3 &tiling_func = op_func_info.GetOpTilingFuncV3();
    const OpParseFuncV3 &parse_func = op_func_info.GetOpParseFuncV3();
    ret = TbeOpTilingPyInterfaceEx3(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                    compile_info_hash, elapse, tiling_func, parse_func);
  } else if (op_func_info.IsFunctionV2()) {
    const OpTilingFuncV2  &tiling_func = op_func_info.GetOpTilingFuncV2();
    ret = TbeOpTilingPyInterfaceEx2New(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                       compile_info_hash, elapse, tiling_func);
  } else if (op_func_info.IsFunctionV1()) {
    const OpTilingFunc  &tiling_func = op_func_info.GetOpTilingFunc();
    ret = TbeOpTilingPyInterfaceEx2BackUp(optype, compile_info, inputs, outputs, run_info_json,
                                          run_info_len, compile_info_hash, elapse, tiling_func);
  } else {
    GE_LOGE("Optiling func of op type[%s] is all empty.", optype);
  }

  return ret;
}

extern "C" int TbeOpTilingPyInterfaceEx(const char *optype, const char *compile_info, const char *inputs,
                                        const char *outputs, char *run_info_json, size_t run_info_len,
                                        uint64_t *elapse) {
  return TbeOpTilingPyInterfaceEx2(optype, compile_info, inputs, outputs, run_info_json, run_info_len, nullptr, elapse);
}

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info, const char *inputs,
                                      const char *outputs, char *run_info_json, size_t run_info_len) {
  return TbeOpTilingPyInterfaceEx(optype, compile_info, inputs, outputs, run_info_json, run_info_len, nullptr);
}
}
