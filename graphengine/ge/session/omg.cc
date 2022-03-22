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

#include "framework/omg/omg.h"
#include <fstream>
#include <iostream>
#include <memory>
#include "common/auth/file_saver.h"
#include "framework/common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge/ge_util.h"
#include "framework/common/helper/model_helper.h"
#include "common/model_parser/model_parser.h"
#include "common/model_saver.h"
#include "common/properties_manager.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/common/params.h"
#include "graph/utils/type_utils.h"
#include "ir_build/option_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/weights_parser.h"
#include "parser/common/pre_checker.h"
#include "parser/common/convert/pb2json.h"
#include "proto/ge_ir.pb.h"
#include "register/op_registry.h"

using nlohmann::json;
using ProcParam = struct PROC_PARAM;
using domi::ModelParserFactory;
using domi::WeightsParserFactory;
using std::ostringstream;

namespace google {
namespace protobuf {
namespace io {
class FileOutputStream;
}
}  // namespace protobuf
}  // namespace google
namespace ge {
namespace {
const std::string kGraphDefaultName = "domi_default";
const std::string kScopeIdAttr = "fusion_scope";
const char *const kOutputTypeSample = "correct sample is \"opname:index:dtype\"";
const char *const kOutputTypeSupport = "only support FP32, FP16, UINT8";
const char *const kOutputTypeError = "The multiple out nodes set in output_type must be found in out_nodes.";
const size_t kNodeNameIndex = 0;
const size_t kIndexStrIndex = 1;
const size_t kDTValueIndex = 2;
const size_t kOmInfoSize = 4;
}  // namespace

// When the model is converted to a JSON file, the following operator attributes in the blacklist will be ignored
const std::set<string> kOmBlackFields = {"output",      "data_offset", "data", "workspace", "workspace_bytes",
                                         "memory_size", "weight_size", "size", "bt",        "quantize_factor"};

static std::map<std::string, ge::DataType> output_type_str_to_datatype = {
    {"FP32", ge::DT_FLOAT}, {"FP16", ge::DT_FLOAT16}, {"UINT8", ge::DT_UINT8}};

static bool CheckInputTrueOrFalse(const std::string &s, const std::string &atc_param) {
  if ((s == "true") || (s == "false")) {
    return true;
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage("E10005", {"parameter", "value"}, {atc_param, s});
    GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--%s]'s value[%s] must be true or false.",
           atc_param.c_str(), s.c_str());
    return false;
  }
}

static void ParseAtcParms(const std::map<std::string, std::string> &atc_params, const std::string &key,
                          std::string &param) {
  auto iter = atc_params.find(key);
  if (iter != atc_params.end()) {
    param = iter->second;
  }
}

static Status CheckInputShapeNode(const ComputeGraphPtr &graph, bool is_dynamic_input,
                                  const std::string &input_shape_range, RunMode run_mode) {
  if (!is_dynamic_input && run_mode != MODEL_TO_JSON && input_shape_range.empty()) {
    for (auto node : graph->GetDirectNode()) {
      if (node->GetType() == DATA) {
        auto data_op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(data_op_desc);
        auto tensor_desc = data_op_desc->MutableInputDesc(0);
        GE_CHECK_NOTNULL(tensor_desc);
        for (auto dim : tensor_desc->GetShape().GetDims()) {
          if (dim < 0) {
            GELOGE(PARAM_INVALID, "[Check][Param]Input op [%s] shape %ld is negative, "
                   "maybe you should set input_shape to specify its shape", node->GetName().c_str(), dim);
            const string reason = "maybe you should set input_shape to specify its shape";
            ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                            {node->GetName(), to_string(dim), reason});
            return PARAM_INVALID;
          }
        }
      }
    }
  }
  for (auto it : domi::GetContext().user_input_dims) {
    std::string node_name = it.first;
    ge::NodePtr node = graph->FindNode(node_name);
    if (node == nullptr) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"}, {"input_shape", node_name});
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_shape]'s opname[%s] is not exist in model",
             node_name.c_str());
      return PARAM_INVALID;
    }
    if (node->GetType() != DATA) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10017", {"parameter", "opname"}, {"input_shape", node_name});
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_shape]'s opname[%s] is not a input opname",
             node_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

void AddAttrsForInputNodes(const vector<string> &adjust_fp16_format_vec, const string &fp16_nodes_name, uint32_t index,
                           OpDescPtr &op_desc) {
  if (AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_DATATYPE, TypeUtils::DataTypeToSerialString(DT_FLOAT16))) {
    if ((index < adjust_fp16_format_vec.size()) && (adjust_fp16_format_vec[index] == "true")) {
      GELOGI("This node [%s] should be set NC1HWC0", fp16_nodes_name.c_str());
      if (!AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_FORMAT, TypeUtils::FormatToSerialString(FORMAT_NC1HWC0))) {
        GELOGW("This node [%s] set NC1HWC0 failed", fp16_nodes_name.c_str());
      }
    }
  }
}

static Status CheckInputFp16Nodes(const ComputeGraphPtr &graph, const string &input_fp16_nodes,
                                  const string &is_input_adjust_hw_layout) {
  GE_CHECK_NOTNULL(graph);
  vector<string> adjust_fp16_format_vec;
  if (!is_input_adjust_hw_layout.empty()) {
    adjust_fp16_format_vec = StringUtils::Split(is_input_adjust_hw_layout, ',');
    for (auto &s : adjust_fp16_format_vec) {
      StringUtils::Trim(s);
      if (!CheckInputTrueOrFalse(s, "is_input_adjust_hw_layout")) {
        GELOGE(PARAM_INVALID, "[Check][Param]Invalid Param, is_input_adjust_hw_layout only support true/false:"
               "but is [%s]", is_input_adjust_hw_layout.c_str());
        return PARAM_INVALID;
      }
    }
  }
  if (input_fp16_nodes.empty()) {
    return SUCCESS;
  }
  GELOGI("The input_fp16_nodes is set %s", input_fp16_nodes.c_str());
  vector<string> input_fp16_nodes_vec = StringUtils::Split(input_fp16_nodes, ';');
  for (uint32_t i = 0; i < input_fp16_nodes_vec.size(); ++i) {
    ge::NodePtr node = graph->FindNode(input_fp16_nodes_vec[i]);
    if (node == nullptr) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"},
                                                      {"input_fp16_nodes", input_fp16_nodes_vec[i]});
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_fp16_nodes]'s opname[%s] is not exist in model",
             input_fp16_nodes_vec[i].c_str());
      return PARAM_INVALID;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->GetType() != DATA) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10017", {"parameter", "opname"},
                                                      {"input_fp16_nodes", input_fp16_nodes_vec[i]});
      GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--input_fp16_nodes]'s opname[%s] is not a input opname",
             input_fp16_nodes_vec[i].c_str());
      return PARAM_INVALID;
    }
    AddAttrsForInputNodes(adjust_fp16_format_vec, input_fp16_nodes_vec[i], i, op_desc);
  }
  return SUCCESS;
}

static Status ParseOutputFp16NodesFormat(const string &is_output_fp16) {
  if (is_output_fp16.empty()) {
    return SUCCESS;
  }

  vector<domiTensorFormat_t> &output_formats = domi::GetContext().output_formats;
  output_formats.clear();
  vector<string> node_format_vec = StringUtils::Split(is_output_fp16, ',');
  for (auto &is_fp16 : node_format_vec) {
    StringUtils::Trim(is_fp16);
    if (!CheckInputTrueOrFalse(is_fp16, "is_output_adjust_hw_layout")) {
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid Param, is_output_adjust_hw_layout "
             "only support true/false: but is [%s]", is_output_fp16.c_str());
      return PARAM_INVALID;
    }
    if (is_fp16 == "false") {
      output_formats.push_back(DOMI_TENSOR_ND);
    } else if (is_fp16 == "true") {
      output_formats.push_back(domi::DOMI_TENSOR_NC1HWC0);
    }
  }
  return SUCCESS;
}

void FindParserSo(const string &path, vector<string> &file_list, string &caffe_parser_path) {
  // path, Change to absolute path
  string real_path = RealPath(path.c_str());
  if (real_path.empty()) {  // plugin path does not exist
    return;
  }
  struct stat stat_buf;
  if ((stat(real_path.c_str(), &stat_buf) != 0) || (!S_ISDIR(stat_buf.st_mode))) {
    GELOGI("The path %s is not a directory.", real_path.c_str());
    return;
  }

  struct dirent *dent(nullptr);
  DIR *dir = opendir(real_path.c_str());

  if (nullptr == dir) {  //  plugin path does not exist
    GELOGW("Open directory %s failed.", path.c_str());
    return;
  }

  while ((dent = readdir(dir)) != nullptr) {
    if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0) continue;
    string name = dent->d_name;
    string full_name = real_path + "/" + name;
    const string so_suff = ".so";
    const string caffe_parser_so_suff = "lib_caffe_parser.so";
    if (name.size() >= so_suff.size() && name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0) {
      if (full_name.size() >= caffe_parser_so_suff.size() &&
          full_name.compare(full_name.size() - caffe_parser_so_suff.size(), caffe_parser_so_suff.size(),
                            caffe_parser_so_suff) == 0) {
        caffe_parser_path = full_name;
      } else {  // save parser so path into file_list vector
        file_list.push_back(full_name);
      }
      continue;
    }

    FindParserSo(full_name, file_list, caffe_parser_path);
  }
  closedir(dir);
  return;
}

Status SetOutFormatAndDataTypeAttr(ge::OpDescPtr op_desc, const ge::Format format, const ge::DataType data_type) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "param op_desc is nullptr, check invalid.");
    GELOGE(domi::FAILED, "[Check][Param]Input op desc invalid.");
    return domi::FAILED;
  }
  (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_NET_OUTPUT_FORMAT, format);
  (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_NET_OUTPUT_DATATYPE, data_type);
  return domi::SUCCESS;
}

bool CheckDigitStr(std::string &str) {
  for (char c : str) {
    if (!isdigit(c)) {
      GELOGE(domi::FAILED, "[Check][Param]value[%s] is not positive integer", str.c_str());
      return false;
    }
  }
  return true;
}

Status StringToInt(std::string &str, int32_t &value) {
  try {
    if (!CheckDigitStr(str)) {
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid of digit string: %s ", str.c_str());
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", str, "is not positive integer"});
      return PARAM_INVALID;
    }
    value = stoi(str);
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "[Check][Param]Invalid of digit string: %s, catch invalid_argument.", str.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"}, {"--output_type", str});
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "[Check][Param]Invalid of digit string: %s, catch out_of_range.", str.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"}, {"--output_type", str});
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status VerifyOutputTypeAndOutNodes(std::vector<std::string> &out_type_vec) {
  std::vector<std::pair<std::string, int32_t>> user_out_nodes = domi::GetContext().user_out_nodes;
  std::set<std::string> out_nodes_info;
  for (uint32_t i = 0; i < user_out_nodes.size(); ++i) {
    // out_nodes set should include output_type and output_format
    std::string tmp = user_out_nodes[i].first + ":" + to_string(user_out_nodes[i].second);
    out_nodes_info.emplace(tmp);
  }
  for (uint32_t i = 0; i < out_type_vec.size(); ++i) {
    if (out_nodes_info.find(out_type_vec[i]) == out_nodes_info.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", out_type_vec[i], kOutputTypeError});
      GELOGE(domi::FAILED, "[Check][Param]Invalid value for --output_type[%s], %s.",
             out_type_vec[i].c_str(), kOutputTypeError);
      return domi::FAILED;
    }
  }
  return domi::SUCCESS;
}

Status CheckOutPutDataTypeSupport(const std::string &output_type) {
  auto it = output_type_str_to_datatype.find(output_type);
  if (it == output_type_str_to_datatype.end()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {"--output_type", output_type, kOutputTypeSupport});
    GELOGE(PARAM_INVALID, "[Check][Param]Invalid value for --output_type[%s], %s.",
           output_type.c_str(), kOutputTypeSupport);
    return domi::FAILED;
  }
  return domi::SUCCESS;
}

Status ParseOutputType(const std::string &output_type, std::map<std::string, vector<std::string>> &output_node_dt_map) {
  if (output_type.find(':') == std::string::npos) {
    GELOGI("output_type is not multiple nodes, means all out nodes");
    return CheckOutPutDataTypeSupport(output_type);
  }
  std::vector<std::string> out_type_vec;
  vector<string> nodes_v = StringUtils::Split(output_type, ';');
  for (const string &node : nodes_v) {
    vector<string> node_index_type_v = StringUtils::Split(node, ':');
    if (node_index_type_v.size() != 3) {  // The size must be 3.
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", node, kOutputTypeSample});
      GELOGE(PARAM_INVALID, "[Parse][Param]Invalid value for --output_type[%s], %s.", node.c_str(), kOutputTypeSample);
      return domi::FAILED;
    }
    ge::DataType tmp_dt;
    std::string node_name = StringUtils::Trim(node_index_type_v[kNodeNameIndex]);
    std::string index_str = StringUtils::Trim(node_index_type_v[kIndexStrIndex]);
    int32_t index;
    if (StringToInt(index_str, index) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Convert][Type]This str must be digit string, while the actual input is %s.",
             index_str.c_str());
      return domi::FAILED;
    }
    std::string dt_value = StringUtils::Trim(node_index_type_v[kDTValueIndex]);
    auto it = output_type_str_to_datatype.find(dt_value);
    if (it == output_type_str_to_datatype.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--output_type", dt_value, kOutputTypeSupport});
      GELOGE(ge::PARAM_INVALID, "[Parse][Param]Invalid value for --output_type[%s], %s.",
             dt_value.c_str(), kOutputTypeSupport);
      return domi::FAILED;
    } else {
      tmp_dt = it->second;
    }
    out_type_vec.push_back(node_name + ":" + index_str);
    std::string index_dt_str = index_str + ":" + TypeUtils::DataTypeToSerialString(tmp_dt);
    auto it1 = output_node_dt_map.find(node_name);
    if (it1 == output_node_dt_map.end()) {
      vector<string> tmp_vec;
      tmp_vec.push_back(index_dt_str);
      output_node_dt_map.emplace(node_name, tmp_vec);
    } else {
      it1->second.push_back(index_dt_str);
    }
  }
  return VerifyOutputTypeAndOutNodes(out_type_vec);
}

Status CheckOutNode(ge::OpDescPtr op_desc, int32_t index) {
  int32_t out_size = op_desc->GetOutputsSize();
  if (index < 0 || index >= out_size) {
    GELOGE(domi::FAILED,
           "[Check][Param]out_node [%s] output index:%d must be smaller "
           "than node output size:%d and can not be negative!",
           op_desc->GetName().c_str(), index, out_size);
    std::string fail_reason = "output index:" + to_string(index) + " must be smaller than output size:" +
                              to_string(out_size) + " and can not be negative!";
    ErrorManager::GetInstance().ATCReportErrMessage("E10003", {"parameter", "value", "reason"},
                                                    {"out_nodes", op_desc->GetName(), fail_reason});
    return domi::FAILED;
  }
  return domi::SUCCESS;
}
Status GetDefaultOutInfo(ge::ComputeGraphPtr &compute_graph,
                         std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  std::vector<std::pair<std::string, int32_t>> default_out_nodes = domi::GetContext().default_out_nodes;
  if (domi::GetContext().type == domi::CAFFE && !default_out_nodes.empty()) {
    for (uint32_t i = 0; i < default_out_nodes.size(); ++i) {
      ge::NodePtr out_node = compute_graph->FindNode(default_out_nodes[i].first);
      if (out_node == nullptr) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"},
                                                        {"out_nodes", default_out_nodes[i].first});
        GELOGE(domi::FAILED, "[Check][Param]Can not find src node (%s) in graph.", default_out_nodes[i].first.c_str());
        return domi::FAILED;
      }
      output_nodes_info.push_back(std::make_pair(out_node, default_out_nodes[i].second));
      GELOGD("Get default output node:%s.", out_node->GetName().c_str());
    }
    return domi::SUCCESS;
  }

  for (ge::NodePtr node : compute_graph->GetDirectNode()) {
    if (!node->GetInAllNodes().empty() && node->GetOutAllNodes().empty()) {
      Status ret = GetOutputLeaf(node, output_nodes_info);
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "find leaf fail.");
    }
  }
  return domi::SUCCESS;
}

Status SetOutputNodeInfo(ge::Graph &graph, const std::string &output_type, const std::string &output) {
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  std::vector<std::pair<std::string, int32_t>> user_out_nodes = domi::GetContext().user_out_nodes;
  std::vector<domiTensorFormat_t> output_formats = domi::GetContext().output_formats;
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_info;
  std::vector<std::string> output_nodes_name;
  std::map<std::string, vector<std::string>> output_node_dt_map;
  if (!output_type.empty()) {
    if (ParseOutputType(output_type, output_node_dt_map) != SUCCESS) {
      GELOGE(domi::FAILED, "[Parse][output_type] failed.");
      return domi::FAILED;
    }
  }

  // User declared outputs
  for (uint32_t i = 0; i < user_out_nodes.size(); ++i) {
    ge::NodePtr out_node = compute_graph->FindNode(user_out_nodes[i].first);
    if (out_node == nullptr) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"},
                                                      {"out_nodes", user_out_nodes[i].first});
      GELOGE(domi::FAILED, "[Check][Param]Can not find src node (%s) in graph.", user_out_nodes[i].first.c_str());
      return domi::FAILED;
    }
    auto op_desc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (CheckOutNode(op_desc, user_out_nodes[i].second) != SUCCESS) {
      GELOGE(domi::FAILED, "[Check][OutNode] (%s) fail.", user_out_nodes[i].first.c_str());
      return domi::FAILED;
    }

    // add user_define_output_nodes attr.
    (void)ge::AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_OUTPUT_NODES, "true");

    if (i < output_formats.size()) {
      if (output_formats[i] == domi::DOMI_TENSOR_NC1HWC0) {
        GELOGI("The output node [%s] should be set NC1HWC0", user_out_nodes[i].first.c_str());
        vector<string> output_fp16_5hd_vec;
        (void)ge::AttrUtils::GetListStr(op_desc, "_user_defined_output_fp16_5hd", output_fp16_5hd_vec);
        output_fp16_5hd_vec.push_back(std::to_string(user_out_nodes[i].second) + ":" + "NC1HWC0");
        (void)ge::AttrUtils::SetListStr(op_desc, "_user_defined_output_fp16_5hd", output_fp16_5hd_vec);
      }
    }
    auto it = output_node_dt_map.find(user_out_nodes[i].first);
    if (it != output_node_dt_map.end()) {
      GELOGI("The output node [%s] need to be set output_type", user_out_nodes[i].first.c_str());
      (void)ge::AttrUtils::SetListStr(op_desc, "_user_defined_output_data_type", it->second);
    }
    output_nodes_info.push_back(std::make_pair(out_node, user_out_nodes[i].second));
  }
  // default output node (leaf)
  if (user_out_nodes.empty()) {
    if (GetDefaultOutInfo(compute_graph, output_nodes_info) != SUCCESS) {
      GELOGE(domi::FAILED, "[Get][DefaultOutInfo] failed.");
      return domi::FAILED;
    }
  }
  GetOutputNodesNameAndIndex(output_nodes_info, output_nodes_name);
  compute_graph->SetGraphOutNodesInfo(output_nodes_info);
  domi::GetContext().net_out_nodes = output_nodes_name;
  return domi::SUCCESS;
}

void GetOutputNodesNameAndIndex(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                                std::vector<std::string> &output_nodes_name) {
  output_nodes_name.clear();
  if (domi::GetContext().out_top_names.empty()) {
    // tf process, no top name.
    for (const auto output_node_info : output_nodes_info) {
      std::string node_name = output_node_info.first->GetName();
      int32_t index = output_node_info.second;
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
    return;
  }
  // caffe process, need add top name after node_name:index
  for (size_t i = 0; i < output_nodes_info.size(); ++i) {
    std::string node_name = output_nodes_info[i].first->GetName();
    int32_t index = output_nodes_info[i].second;
    if (i < domi::GetContext().out_top_names.size()) {
      output_nodes_name.push_back(node_name + ":" + std::to_string(index) + ":" + domi::GetContext().out_top_names[i]);
    } else {
      GELOGW("Get top name of node [%s] fail.", node_name.c_str());
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
  }
}

Status GetOutputLeaf(NodePtr node, std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  ge::OpDescPtr tmpDescPtr = node->GetOpDesc();
  if (tmpDescPtr == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node has no opdesc.");
    GELOGE(domi::FAILED, "[Check][Param]Get outnode op desc fail.");
    return domi::FAILED;
  }
  size_t size = tmpDescPtr->GetOutputsSize();
  if (node->GetType() != NETOUTPUT) {
    for (size_t index = 0; index < size; ++index) {
      output_nodes_info.push_back(std::make_pair(node, index));
      GELOGD("Get output leaf node:%s.", node->GetName().c_str());
    }
  } else {
    const auto in_anchors = node->GetAllInDataAnchors();
    for (auto in_anchor : in_anchors) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr) {
        REPORT_INNER_ERROR("E19999", "GetPeerOutAnchor return nullptr, node:%s.", node->GetName().c_str());
        GELOGE(domi::FAILED, "[Invoke][GetPeerOutAnchor]Get leaf node op desc fail.");
        return domi::FAILED;
      }
      auto out_node = out_anchor->GetOwnerNode();
      output_nodes_info.push_back(std::make_pair(out_node, out_anchor->GetIdx()));
    }
  }
  return SUCCESS;
}

///
/// @ingroup domi_common
/// @brief Initialize omgcontext based on command line input
/// @param [in] input_shape Input shape string to be parsed
/// @return SUCCESS: parse successfully; PARAM_INVALID：parse failed
///
Status InitDomiOmgContext(const string &input_shape, const string &input_format, const string &net_format,
                          bool is_dynamic_input) {
  // Clear omgcontext data first
  domi::GetContext().input_dims.clear();
  domi::GetContext().user_input_dims.clear();
  domi::GetContext().is_dynamic_input = is_dynamic_input;

  // the default value is ND
  domi::GetContext().format = DOMI_TENSOR_ND;
  if (!input_format.empty()) {
    auto iter = ge::input_format_str_to_geformat.find(input_format);
    if (iter != ge::input_format_str_to_geformat.end()) {
      domi::GetContext().format = iter->second;
    } else {
      REPORT_INNER_ERROR("E19999", "param input_format:%s is not support, "
                         "expect ND/NCHW/NHWC/CHWN/NC1HWC0/NHWC1C0.", input_format.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param]Input format %s not support, "
             "expect ND/NCHW/NHWC/CHWN/NC1HWC0/NHWC1C0.", input_format.c_str());
      return PARAM_INVALID;
    }
  }

  // Input is empty, do not process
  if (input_shape.empty()) {
    return SUCCESS;
  }

  // Analyze the input shape paramete
  map<string, vector<int64_t>> &shape_map = domi::GetContext().input_dims;

  if (!ge::ParseInputShape(input_shape, domi::GetContext().input_dims, domi::GetContext().user_input_dims,
                           is_dynamic_input) || shape_map.empty()) {
    REPORT_CALL_ERROR("E19999", "ParseInputShape failed for %s", input_shape.c_str());
    GELOGE(PARAM_INVALID, "[Parse][InputShape] %s failed.", input_shape.c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status ParseOutNodes(const string &out_nodes) {
  try {
    // parse output node
    if (!out_nodes.empty()) {
      domi::GetContext().out_nodes_map.clear();
      domi::GetContext().user_out_nodes.clear();
      domi::GetContext().user_out_nodes_top_vec.clear();

      vector<string> nodes_v = StringUtils::Split(out_nodes, ';');
      for (const string &node : nodes_v) {
        vector<string> key_value_v = StringUtils::Split(node, ':');
        if (key_value_v.size() != 2) {  // The size must be 2.
          if (key_value_v.size() == 1 && domi::GetContext().type == domi::CAFFE) {
            domi::GetContext().user_out_nodes_top_vec.push_back(node);
            continue;
          }
          ErrorManager::GetInstance().ATCReportErrMessage(
              "E10001", {"parameter", "value", "reason"},
              {"--out_nodes", node, "the correct format is \"node_name1:0;node_name1:1;node_name2:0\""});
          GELOGE(PARAM_INVALID,
                 "[Parse][Param]The input format of --out_nodes is invalid, the correct format is "
                 "\"node_name1:0;node_name1:1;node_name2:0\", while the actual input is %s.",
                 node.c_str());
          return PARAM_INVALID;
        }
        if (!domi::GetContext().user_out_nodes_top_vec.empty()) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                          {"--out_nodes", out_nodes, "is not all index or top_name"});
          GELOGE(PARAM_INVALID, "[Parse][Param]This out_nodes str must be all index or top_name, "
                 "while the actual input is %s", out_nodes.c_str());
          return PARAM_INVALID;
        }
        // stoi: The method may throw an exception: invalid_argument/out_of_range
        if (!CheckDigitStr(key_value_v[1])) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                          {"--out_nodes", out_nodes, "index is not positive integer"});
          GELOGE(PARAM_INVALID, "[Parse][Param]This str must be digit string, while the actual input is %s",
                 out_nodes.c_str());
          return PARAM_INVALID;
        }

        auto iter = domi::GetContext().out_nodes_map.find(key_value_v[0]);
        int32_t index = stoi(StringUtils::Trim(key_value_v[1]));
        GELOGD("Get output info: node[%s] and index[%d]", key_value_v[0].c_str(), index);
        if (iter != domi::GetContext().out_nodes_map.end()) {
          iter->second.emplace_back(index);
        } else {
          std::vector<int32_t> index_v;
          index_v.emplace_back(index);
          domi::GetContext().out_nodes_map.emplace(key_value_v[0], index_v);
        }
        domi::GetContext().user_out_nodes.push_back(std::make_pair(key_value_v[0], index));
      }
    }
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "[Parse][Param]Invalid of out_nodes: %s ", out_nodes.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"}, {"--out_nodes", out_nodes});
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "[Parse][Param]Invalid of out_nodes: %s ", out_nodes.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"}, {"--out_nodes", out_nodes});
    return PARAM_INVALID;
  }
  return SUCCESS;
}

/// @ingroup domi_common
///  @brief Judge whether the op_Name_Map parameter matches the network
///  @param [in] graph Input network graph
///  @return SUCCESS: Input parameters are correct; PARAM_INVALID: Input parameters are incorrect
///
static Status CheckOpNameMap(const ComputeGraphPtr &graph, const std::string &op_conf) {
  GE_CHECK_NOTNULL(graph);
  map<string, string> graphNodeTypes;
  for (const NodePtr &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "param graph's node has no opdesc.");
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid parameter for opDesc.");
      return PARAM_INVALID;
    }
    graphNodeTypes[op_desc->GetType()] = "";
  }
  std::map<std::string, std::string> &propertiesMap = domi::GetContext().op_conf_map;
  if (propertiesMap.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10003", {"parameter", "value", "reason"}, {"op_name_map", op_conf, "the file content is empty"});
    GELOGE(PARAM_INVALID, "[Check][Param]op_name_map file content is empty, please check file!");
    return PARAM_INVALID;
  }
  for (auto iter = propertiesMap.begin(); iter != propertiesMap.end(); iter++) {
    GE_IF_BOOL_EXEC(graphNodeTypes.find(iter->second) == graphNodeTypes.end(),
                    ErrorManager::GetInstance().ATCReportErrMessage(
                        "E10003", {"parameter", "value", "reason"},
                        {"op_name_map", op_conf, "type[" + iter->second + "] is not found in model"});
                    GELOGE(PARAM_INVALID, "[Find][NodeType]Invalid parameter for op_name_map.");
                    return PARAM_INVALID;);
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY Status ParseGraph(ge::Graph &graph, const std::map<string, string> &atc_params,
                                           const char *model_file, const char *weights_file, domi::FrameworkType type,
                                           const char *op_conf, const char *target, RunMode run_mode,
                                           bool is_dynamic_input) {
  GE_CHECK_NOTNULL(model_file);
  GE_CHECK_NOTNULL(weights_file);
  domi::GetContext().type = type;
  domi::GetContext().run_mode = run_mode;
  // Prevent data residue in multiple calls
  PreChecker::Instance().Clear();

  Params::Instance()->SetTarget(target);

  // Create an empty computegraph
  std::string om_name;
  ParseAtcParms(atc_params, "output", om_name);
  ModelHelper model_helper;
  string graph_name = "";
  Status name_ret = model_helper.GetBaseNameFromFileName(om_name, graph_name);
  if (name_ret != SUCCESS) {
    graph_name = kGraphDefaultName + "_" + CurrentTimeInStr();
  }
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(compute_graph);
  graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  // initialize omgContext
  std::string input_shape;
  ParseAtcParms(atc_params, "input_shape", input_shape);
  std::string input_format;
  ParseAtcParms(atc_params, "input_format", input_format);
  GE_RETURN_WITH_LOG_IF_ERROR(InitDomiOmgContext(input_shape, input_format, "", is_dynamic_input),
                              "[Call][InitDomiOmgContext] ret fail");

  std::string is_output_adjust_hw_layout;
  ParseAtcParms(atc_params, "is_output_adjust_hw_layout", is_output_adjust_hw_layout);
  GE_RETURN_WITH_LOG_IF_ERROR(ParseOutputFp16NodesFormat(is_output_adjust_hw_layout),
                              "[Call][ParseOutputFp16NodesFormat]Parse is_output_fp16 failed");

  std::string out_nodes;
  ParseAtcParms(atc_params, "out_nodes", out_nodes);
  GE_RETURN_WITH_LOG_IF_ERROR(ParseOutNodes(out_nodes), "[Parse][OutNodes] fail");

  std::string output_type;
  ParseAtcParms(atc_params, "output_type", output_type);

  // parse configuration item
  if (op_conf != nullptr && *op_conf != '\0') {
    // divided by ":"
    PropertiesManager::Instance().SetPropertyDelimiter(OP_CONF_DELIMITER);
    // Parsing the op_conf configuration item file
    GE_IF_BOOL_EXEC(!PropertiesManager::Instance().Init(op_conf),
                    ErrorManager::GetInstance().ATCReportErrMessage("E10003", {"parameter", "value", "reason"},
                                                                    {"op_name_map", op_conf, "file content error"});
                    GELOGE(FAILED, "[Invoke][Init]op_name_map init failed!");
                    return FAILED);
    // Return map and put it into ATC global variable
    domi::GetContext().op_conf_map = PropertiesManager::Instance().GetPropertyMap();
  }

  // parse network model
  auto model_parser = ModelParserFactory::Instance()->CreateModelParser(type);
  if (model_parser == nullptr) {
    REPORT_INNER_ERROR("E19999", "CreateModelParser failed, type:%d", type);
    GELOGE(FAILED, "[Create][ModelParser] ret fail, type:%d.", type);
    return FAILED;
  }
  UpdateParserCtxWithOmgCtx();
  Status ret = model_parser->Parse(model_file, graph);
  UpdateOmgCtxWithParserCtx();

  // Generate the report in case of pre inspection failure or only pre inspection mode
  if (PreChecker::Instance().HasError() || run_mode == ONLY_PRE_CHECK) {
    std::string check_report;
    ParseAtcParms(atc_params, "check_report", check_report);
    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().Save(check_report),
                                "[Invoke][Save]Generate pre-checking report failed.");
    GEEVENT("The pre-checking report has been saved to %s.", check_report.c_str());
  }

  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "ATC model parse ret fail.");

  std::string input_fp16_nodes;
  ParseAtcParms(atc_params, "input_fp16_nodes", input_fp16_nodes);
  std::string is_input_adjust_hw_layout;
  ParseAtcParms(atc_params, "is_input_adjust_hw_layout", is_input_adjust_hw_layout);
  compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_RETURN_IF_ERROR(CheckInputFp16Nodes(compute_graph, input_fp16_nodes, is_input_adjust_hw_layout));
  std::string input_shape_range;
  ParseAtcParms(atc_params, INPUT_SHAPE_RANGE, input_shape_range);
  GE_RETURN_IF_ERROR(CheckInputShapeNode(compute_graph, is_dynamic_input, input_shape_range, run_mode));

  // Verify the contents of the op_name_map
  if (op_conf != nullptr && *op_conf != '\0') {
    GE_RETURN_WITH_LOG_IF_ERROR(CheckOpNameMap(compute_graph, op_conf),
                                "[Invoke][CheckOpNameMap]op_name_map parameter is not fit with input net!");
  }

  // Print parse network structure
  compute_graph->Dump();

  // parse weight
  graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  auto weights_parser = WeightsParserFactory::Instance()->CreateWeightsParser(type);
  ret = weights_parser->Parse(weights_file, graph);

  // IN ONLY_PRE_CHECK mode, generate pre inspection report only.
  if (PreChecker::Instance().HasError() || run_mode == ONLY_PRE_CHECK) {
    std::string check_report;
    ParseAtcParms(atc_params, "check_report", check_report);
    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().Save(check_report),
                                "[Invoke][Save]Generate pre-checking report failed.");
    GEEVENT("The pre-checking report has been saved to %s.", check_report.c_str());
  }
  // Prevent data residue in multiple calls
  PreChecker::Instance().Clear();

  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "[Check][State]ATC weights parse ret fail.");

  // parser input shape range and update op shape range
  GE_RETURN_WITH_LOG_IF_ERROR(UpdateDynamicInputShapeRange(compute_graph, input_shape_range),
                              "[Update][DynamicInputShapeRange] failed");

  GELOGI("ATC parser success.");

  return SUCCESS;
}

void GetGroupName(ge::proto::ModelDef &model_def) {
  auto modelAttrMap = model_def.mutable_attr();
  auto fusionModelOpListIter = modelAttrMap->find(MODEL_ATTR_FUSION_MODEL_DEF);
  GE_IF_BOOL_EXEC(
      fusionModelOpListIter != modelAttrMap->end(), int fusionOpIndex = 0;
      for (int i = 0; i < model_def.graph_size(); i++) {
        auto graph = model_def.mutable_graph(i);
        for (int j = 0; j < graph->op_size(); j++) {
          int64_t scope_id = 0;
          auto bt = fusionModelOpListIter->second.list().bt(fusionOpIndex++);
          ge::proto::OpDef fusion_op_def;
          GE_CHK_BOOL_EXEC(bt.size() != 0, GELOGW("Invalid bt size"); return;);

          (void)(fusion_op_def.ParseFromArray(bt.data(), bt.size()));
          auto fusion_attr_map = fusion_op_def.mutable_attr();
          auto fusion_iter = fusion_attr_map->find(kScopeIdAttr);
          GE_IF_BOOL_EXEC(fusion_iter == fusion_attr_map->end(), continue;);

          scope_id = fusion_iter->second.i();
          ge::proto::OpDef *opdef = graph->mutable_op(j);
          auto attr_map = opdef->mutable_attr();

          int64_t stream_id = opdef->stream_id();

          uint16_t l1_id = (((uint64_t)scope_id & 0xFFFF0000)) >> 16;
          GE_IF_BOOL_EXEC(l1_id != 0, ostringstream groupName; groupName << "group_op_l1_" << l1_id << "_" << stream_id;
                          (*attr_map)["group_op_name"].set_s(groupName.str()); continue;);

          uint16_t ub_id = ((uint64_t)scope_id & 0xFFFF);
          GE_IF_BOOL_EXEC(ub_id != 0, ostringstream groupName; groupName << "group_op_ub_" << ub_id << "_" << stream_id;
                          (*attr_map)["group_op_name"].set_s(groupName.str()););
        }
      });
}

FMK_FUNC_HOST_VISIBILITY void PrintModelInfo(ge::proto::ModelDef *model_def, uint32_t modeldef_size) {
  std::cout << "============ Display Model Info start ============" << std::endl;

  auto model_attr_map = model_def->mutable_attr();
  // system info
  auto iter = model_attr_map->find(ATTR_MODEL_ATC_VERSION);
  auto atc_version = (iter != model_attr_map->end()) ? iter->second.s() : "";
  iter = model_attr_map->find("soc_version");
  auto soc_version = (iter != model_attr_map->end()) ? iter->second.s() : "";
  iter = model_attr_map->find("framework_type");
  auto framework_type = (iter != model_attr_map->end()) ? iter->second.s() : "";
  // original atc cmdline
  iter = model_attr_map->find(ATTR_MODEL_ATC_CMDLINE);
  auto cmdline = (iter != model_attr_map->end()) ? iter->second.s() : "";
  std::cout << "Original Atc command line: "
            << cmdline << std::endl
            << "system   info: "
            <<  ATTR_MODEL_ATC_VERSION
            << "[" << atc_version << "], "
            << "soc_version"
            << "[" << soc_version << "], "
            << "framework_type"
            << "[" << framework_type << "]." << std::endl;

  // resource info
  iter = model_attr_map->find(ATTR_MODEL_MEMORY_SIZE);
  auto memory_size = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  iter = model_attr_map->find(ATTR_MODEL_WEIGHT_SIZE);
  auto weight_size = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  iter = model_attr_map->find(ATTR_MODEL_STREAM_NUM);
  auto stream_num = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  iter = model_attr_map->find(ATTR_MODEL_EVENT_NUM);
  auto event_num = (iter != model_attr_map->end()) ? iter->second.i() : -1;
  std::cout << "resource info: "
            << ATTR_MODEL_MEMORY_SIZE
            << "[" << memory_size << " B], "
            << ATTR_MODEL_WEIGHT_SIZE
            << "[" << weight_size << " B], "
            << ATTR_MODEL_STREAM_NUM
            << "[" << stream_num << "], "
            << ATTR_MODEL_EVENT_NUM
            << "[" << event_num << "]."
            << std::endl;

  // om info
  iter = model_attr_map->find("om_info_list");
  if (iter == model_attr_map->end()) {
    std::cout << "Display Model Info failed, attr \"om_info_list\" is not found in om, check the version is matched."
              << std::endl;
    std::cout << "============ Display Model Info end   ============"  << std::endl;
    return;
  }
  auto list_size = iter->second.list().i_size();
  if (list_size == kOmInfoSize) {
    std::cout << "om       info: "
              << "modeldef_size"
              << "[" << modeldef_size << " B], "
              << "weight_data_size"
              << "[" << iter->second.list().i(0) << " B], "
              << "tbe_kernels_size"
              << "[" << iter->second.list().i(1) << " B], "
              << "cust_aicpu_kernel_store_size"
              << "[" << iter->second.list().i(2) << " B], "
              << "task_info_size"
              << "[" << iter->second.list().i(3) << " B]." << std::endl;
  } else {
    std::cout << "Display Model Info error, please check!"  << std::endl;
  };

  std::cout << "============ Display Model Info end   ============"  << std::endl;
}

FMK_FUNC_HOST_VISIBILITY Status ConvertOm(const char *model_file, const char *json_file, bool is_covert_to_json) {
  GE_CHECK_NOTNULL(model_file);
  if (is_covert_to_json) {
    GE_CHECK_NOTNULL(json_file);
  }
  ge::ModelData model;

  // Mode 2 does not need to verify the priority, and a default value of 0 is passed
  int32_t priority = 0;

  // Load model from file
  Status ret = ModelParserBase::LoadFromFile(model_file, priority, model);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "LoadFromFile failed, file:%s", model_file);
    GELOGE(ret, "[Invoke][LoadFromFile] failed.");
    return ret;
  }

  uint8_t *model_data = nullptr;
  uint32_t model_len = 0;
  try {
    // Parse the contents of the file to get the modeldef object
    ret = ModelParserBase::ParseModelContent(model, model_data, model_len);
    if (ret == SUCCESS) {
      OmFileLoadHelper omFileLoadHelper;
      ge::graphStatus status = omFileLoadHelper.Init(model_data, model_len);
      if (status != ge::GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Om file:%s init failed", model_file);
        GELOGE(ge::FAILED, "[Invoke][Init]Om file init failed.");
        if (model.model_data != nullptr) {
          delete[] reinterpret_cast<char *>(model.model_data);
          model.model_data = nullptr;
        }
        return status;
      }

      ModelPartition ir_part;
      status = omFileLoadHelper.GetModelPartition(MODEL_DEF, ir_part);
      if (status != ge::GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Get model part of om file:%s failed", model_file);
        GELOGE(ge::FAILED, "[Get][ModelPartition] failed.");
        if (model.model_data != nullptr) {
          delete[] reinterpret_cast<char *>(model.model_data);
          model.model_data = nullptr;
        }
        return status;
      }

      ge::proto::ModelDef model_def;

      // De serialization
      bool flag = ReadProtoFromArray(ir_part.data, ir_part.size, &model_def);
      if (flag) {
        if (is_covert_to_json) {
          GetGroupName(model_def);

          json j;
          Pb2Json::Message2Json(model_def, kOmBlackFields, j, true);

          ret = ModelSaver::SaveJsonToFile(json_file, j);
        } else {
          PrintModelInfo(&model_def, ir_part.size);
        }
      } else {
        ret = INTERNAL_ERROR;
        REPORT_CALL_ERROR("E19999", "ReadProtoFromArray failed for om file:%s", model_file);
        GELOGE(ret, "[Read][Proto]From Array failed.");
      }
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage("E10003",
          {"parameter", "value", "reason"}, {"om", model_file, "invalid om file, can't be parsed"});
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[Parse][ModelContent] failed because of invalid om file. Please check --om param.");
    }

    if (model.model_data != nullptr) {
      delete[] reinterpret_cast<char *>(model.model_data);
      model.model_data = nullptr;
    }
    return ret;
  } catch (const std::exception &e) {
    REPORT_INNER_ERROR("E19999", "Convert om model to json failed, exception message:%s, model_file:%s",
                       std::string(e.what()).c_str(), model_file);
    GELOGE(FAILED, "[Save][Model]Convert om model to json failed, exception message : %s.", e.what());
    return FAILED;
  }
}

FMK_FUNC_HOST_VISIBILITY Status ConvertPbtxtToJson(const char *model_file, const char *json_file) {
  ge::ModelData model;
  // Mode 2 does not need to verify the priority, and a default value of 0 is passed
  int32_t priority = 0;
  // Load model from file
  Status ret = ModelParserBase::LoadFromFile(model_file, priority, model);
  auto free_model_data = [](void **ptr) -> void {
    if (ptr != nullptr && *ptr != nullptr) {
      delete[] reinterpret_cast<char *>(*ptr);
      *ptr = nullptr;
    }
  };
  if (ret != SUCCESS) {
    free_model_data(&model.model_data);
    REPORT_CALL_ERROR("E19999", "LoadFromFile failed.");
    GELOGE(ret, "[Invoke][LoadFromFile] failed.");
    return ret;
  }

  try {
    bool flag = false;
    ge::proto::ModelDef model_def;
    flag = google::protobuf::TextFormat::ParseFromString(reinterpret_cast<char *>(model.model_data), &model_def);

    if (!flag) {
      free_model_data(&model.model_data);
      REPORT_CALL_ERROR("E19999", "ParseFromString failed for model_file:%s", model_file);
      GELOGE(FAILED, "[Invoke][ParseFromString] failed.");
      return FAILED;
    }
    GetGroupName(model_def);
    json j;
    Pb2Json::Message2Json(model_def, kOmBlackFields, j, true);
    ret = ModelSaver::SaveJsonToFile(json_file, j);
    if (ret != SUCCESS) {
      free_model_data(&model.model_data);
      REPORT_CALL_ERROR("E19999", "SaveJsonToFile failed.");
      GELOGE(ret, "[Save][Json] to file fail.");
      return ret;
    }
    free_model_data(&model.model_data);
    return SUCCESS;
  } catch (google::protobuf::FatalException &e) {
    free_model_data(&model.model_data);
    REPORT_INNER_ERROR("E19999", "ParseFromString failed, exception message:%s, model_file:%s",
                       std::string(e.what()).c_str(), model_file);
    GELOGE(FAILED, "[Invoke][ParseFromString] failed. exception message : %s", e.what());
    return FAILED;
  } catch (const std::exception &e) {
    REPORT_INNER_ERROR("E19999", "ParseFromString failed, exception message:%s, model_file:%s",
                       std::string(e.what()).c_str(), model_file);
    GELOGE(FAILED, "[Save][pbtxt]Convert pbtxt to json failed, exception message : %s.", e.what());
    return FAILED;
  }
}

FMK_FUNC_HOST_VISIBILITY Status ConvertFwkModelToJson(const domi::FrameworkType framework, const char *model_file,
                                                      const char *json_file) {
  if (framework == domi::CAFFE || framework == domi::TENSORFLOW || framework == domi::ONNX) {
    auto model_parser = ModelParserFactory::Instance()->CreateModelParser(framework);
    if (model_parser == nullptr) {
      REPORT_INNER_ERROR("E19999", "CreateModelParser failed, framework:%d.", framework);
      GELOGE(FAILED, "[Create][ModelParser] ret fail, framework:%d.", framework);
      return FAILED;
    }
    return model_parser->ToJson(model_file, json_file);
  }

  ErrorManager::GetInstance().ATCReportErrMessage(
      "E10001", {"parameter", "value", "reason"},
      {"--framework", std::to_string(framework), "only support 0(Caffe) 3(TensorFlow) 5(Onnx) when model set 1"});
  GELOGE(PARAM_INVALID, "[Check][Param]Input parameter[--framework] is mandatory "
         "and it's value must be: 0(Caffe) 3(TensorFlow) or 5(Onnx).");
  return PARAM_INVALID;
}

FMK_FUNC_HOST_VISIBILITY Status DumpInfershapeJson(const ge::Graph &graph, const char *json_file) {
  // Create buffer
  GELOGI("Enter to dump infershape json schedule.");
  ge::Model model("", "");
  model.SetGraph(graph);
  Buffer buffer;
  model.Save(buffer, true);

  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    std::string str(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      REPORT_CALL_ERROR("E19999", "ParseFromString failed.");
      GELOGE(GRAPH_FAILED, "[Invoke][ParseFromString] failed.");
      return FAILED;
    }

    nlohmann::json j;
    Pb2Json::Message2Json(ge_proto, std::set<string>(), j);

    ModelSaver::SaveJsonToFile(json_file, j);
  }
  return SUCCESS;
}

void UpdateOmgCtxWithParserCtx() {
  domi::GetContext().format = GetParserContext().format;
  domi::GetContext().input_dims = GetParserContext().input_dims;
  domi::GetContext().user_input_dims = GetParserContext().user_input_dims;
  domi::GetContext().is_dynamic_input = GetParserContext().is_dynamic_input;
  domi::GetContext().type = GetParserContext().type;
  domi::GetContext().user_out_nodes = GetParserContext().user_out_nodes;
  domi::GetContext().train_flag = GetParserContext().train_flag;
  domi::GetContext().run_mode = GetParserContext().run_mode;
  domi::GetContext().op_conf_map = GetParserContext().op_conf_map;
  domi::GetContext().out_nodes_map = GetParserContext().out_nodes_map;
  domi::GetContext().input_nodes_format_map = GetParserContext().input_nodes_format_map;
  domi::GetContext().out_top_names = GetParserContext().out_top_names;
  domi::GetContext().user_out_nodes_top_vec = GetParserContext().user_out_nodes_top_vec;
  domi::GetContext().default_out_nodes = GetParserContext().default_out_nodes;
  domi::GetContext().data_top_names = GetParserContext().data_top_names;
}

void UpdateParserCtxWithOmgCtx() {
  GetParserContext().format = domi::GetContext().format;
  GetParserContext().input_dims = domi::GetContext().input_dims;
  GetParserContext().user_input_dims = domi::GetContext().user_input_dims;
  GetParserContext().is_dynamic_input = domi::GetContext().is_dynamic_input;
  GetParserContext().type = domi::GetContext().type;
  GetParserContext().user_out_nodes = domi::GetContext().user_out_nodes;
  GetParserContext().train_flag = domi::GetContext().train_flag;
  GetParserContext().run_mode = domi::GetContext().run_mode;
  GetParserContext().op_conf_map = domi::GetContext().op_conf_map;
  GetParserContext().out_nodes_map = domi::GetContext().out_nodes_map;
  GetParserContext().input_nodes_format_map = domi::GetContext().input_nodes_format_map;
  GetParserContext().out_top_names = domi::GetContext().out_top_names;
  GetParserContext().user_out_nodes_top_vec = domi::GetContext().user_out_nodes_top_vec;
  GetParserContext().data_top_names = domi::GetContext().data_top_names;
}
}  // namespace ge
