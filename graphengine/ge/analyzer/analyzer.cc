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

#include "analyzer/analyzer.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
using json = nlohmann::json;
using Status = ge::Status;
using ComputeGraph = ge::ComputeGraph;
using namespace analyzer;

namespace {
constexpr int kFileAuthority = 0640;
constexpr int kJsonDumpLevel = 4;

const std::string kFilePath = "./";
const std::string kAnalyzeFile = "ge_check_op.json";

const std::string kUnknownShape = "unknownshape";
const std::string kUnsupport    = "unsupport";

const std::string kSessionId = "session_id";
const std::string kGraphId   = "graph_id";
const std::string kOpInfo    = "op_info";
const std::string kErrorType = "error_type";
const std::string kOpName    = "name";
const std::string kOpType    = "type";
const std::string kReason    = "reason";
const std::string kInput     = "input";
const std::string kOutput    = "output";
const std::string kShape     = "shape";
const std::string kDataType  = "data_type";
const std::string kLayout    = "layout";
const std::string kResult    = "result";
const std::string kOp        = "op";

std::map<analyzer::AnalyzeType, std::string> errors_map {
  {PARSER,         "paser_error"},
  {INFER_SHAPE,    "infer_shape_error"},
  {CHECKSUPPORT,   "check_support_error"},
  {GRAPH_OPTIMIZE, "graph_optimize_error"},
  {GRAPH_PARTION,  "graph_partion_error"},
  {GRAPH_BUILDER,  "graph_builder_error"}
};
}

Analyzer* Analyzer::GetInstance() {
  static Analyzer instance;
  return &instance;
}

Status Analyzer::BuildJsonObject(uint64_t session_id, uint64_t graph_id) {
  GELOGD("Start to build map. SessionId:%lu GraphId:%lu", session_id, graph_id);
  std::lock_guard<std::recursive_mutex> lg(mutex_);
  auto iter = graph_infos_.find(session_id);
  if (iter == graph_infos_.end()) {
    std::shared_ptr<GraphInfo> graph_info(new(std::nothrow) GraphInfo());
    GE_CHECK_NOTNULL(graph_info);
    std::map<uint64_t, std::shared_ptr<GraphInfo>> graph_map;
    graph_map[graph_id] = graph_info;
    graph_info->session_id = session_id;
    graph_info->graph_id = graph_id;
    graph_infos_.insert({session_id, graph_map});
  } else {
    auto iter1 = (iter->second).find(graph_id);
    if (iter1 == (iter->second).end()) {
      std::shared_ptr<GraphInfo> graph_info(new(std::nothrow) GraphInfo());
      GE_CHECK_NOTNULL(graph_info);
      graph_info->session_id = session_id;
      graph_info->graph_id = graph_id;
      (iter->second).insert({graph_id, graph_info});
    } else {
      GELOGI("session_id:%lu graph_id:%lu already existed json object", session_id, graph_id);
    }
  }
  return SUCCESS;
}

ge::Status Analyzer::Initialize() {
  // Initialize file
  string real_path = RealPath(kFilePath.c_str());
  if (real_path.empty()) {
    GELOGE(FAILED, "[Check][AnalyzeFilePath]File path is empty, Path invalid.");
    REPORT_CALL_ERROR("E19999", "Analyze file path check invalid, it is empty");
    return FAILED;
  }
  json_file_name_ = real_path + "/" + kAnalyzeFile;

  return SUCCESS;
}

void Analyzer::Finalize() {
  GELOGD("Analyzer start to finalize!");
  std::lock_guard<std::recursive_mutex> lg(mutex_);
  for (auto &session_resource : graph_infos_) {
    session_resource.second.clear();
  }
  graph_infos_.clear();

  std::lock_guard<std::mutex> lk(file_mutex_);
  if (json_file_.is_open()) {
    json_file_.close();
  }
}

void Analyzer::DestroySessionJsonObject(uint64_t session_id) {
  std::lock_guard<std::recursive_mutex> lg(mutex_);
  auto iter = graph_infos_.find(session_id);
  if (iter == graph_infos_.end()) {
    GELOGW("can not find the stored object by session_id[%lu].Do nothing", session_id);
  } else {
    graph_infos_.erase(iter);
  }
}

void Analyzer::DestroyGraphJsonObject(uint64_t session_id, uint64_t graph_id) {
  std::lock_guard<std::recursive_mutex> lg(mutex_);
  auto iter = graph_infos_.find(session_id);
  if (iter == graph_infos_.end()) {
    GELOGW("can not find the stored object by session_id[%lu].Do nothing", session_id);
  } else {
    auto iter1 = (iter->second).find(graph_id);
    if (iter1 == (iter->second).end()) {
      GELOGW("Can not find the graph json object by session_id[%lu] and graph_id[%lu]. Do nothing.", session_id,
             graph_id);
      return;
    }
    (iter->second).erase(iter1);
  }
}

std::shared_ptr<GraphInfo> Analyzer::GetJsonObject(uint64_t session_id, uint64_t graph_id) {
  std::lock_guard<std::recursive_mutex> lg(mutex_);
  auto iter = graph_infos_.find(session_id);
  if (iter == graph_infos_.end()) {
    GELOGE(PARAM_INVALID, "[Check][SessionId]session_id:%lu does not exist! "
           "graph_id:%lu", session_id, graph_id);
    REPORT_INNER_ERROR("E19999", "Sessin_id %lu does not exist, graph_id %lu",
                       session_id, graph_id);
    return nullptr;
  } else {
    auto iter1 = (iter->second).find(graph_id);
    if (iter1 == (iter->second).end()) {
      GELOGE(PARAM_INVALID, "[Check][GraphId]graph_id:%lu does not exist! "
             "session_id:%lu.", graph_id, session_id);
      REPORT_INNER_ERROR("E19999", "Graph_id %lu does not exist, session_id %lu",
                         graph_id, session_id);
      return nullptr;
    }
    GELOGI("GetJsonObject Success!session_id:%lu graph_id:%lu", session_id, graph_id);
    return iter1->second;
  }
}

void Analyzer::ClearHistoryFile() {
  GELOGD("Analyzer start to clear history file!");

  // Remove history files
  int res = remove(json_file_name_.c_str());
  GELOGD("remove file %s, result:%d", json_file_name_.c_str(), res);
}

ge::Status Analyzer::CreateAnalyzerFile() {
  if (is_json_file_create_) {
    GELOGD("analyzer file has been created!No necessary to create again!");
    return SUCCESS;
  }
  GELOGD("start to create analyzer file!");

  std::lock_guard<std::mutex> lg(file_mutex_);
  int fd = open(json_file_name_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, kFileAuthority);
  if (fd < 0) {
    GELOGE(INTERNAL_ERROR, "[FileOpen][AnalyzeFile]Fail to open the analyze file: %s.",
           json_file_name_.c_str());
    REPORT_INNER_ERROR("E19999", "Failed to open analyze file %s", json_file_name_.c_str());
    return INTERNAL_ERROR;
  }
  if (close(fd) != 0) {
    GELOGE(INTERNAL_ERROR, "[FileClose][AnalyzeFile]Fail to close the analyze file: %s.",
           json_file_name_.c_str());
    REPORT_INNER_ERROR("E19999", "Failed to clsoe analyze file %s", json_file_name_.c_str());
    return INTERNAL_ERROR;
  }
  is_json_file_create_ = true;

  GELOGD("success to create analyzer file[%s]!", json_file_name_.c_str());
  return SUCCESS;
}

ge::Status Analyzer::SaveAnalyzerDataToFile(uint64_t session_id, uint64_t graph_id) {
  GELOGD("start to save analyze file");

  auto graph_info = GetJsonObject(session_id, graph_id);
  GE_CHECK_NOTNULL(graph_info);
  if (graph_info->op_info.size() == 0) {
    GELOGD("session_id:%lu graph_id:%lu does not owner op info, break it!", session_id, graph_id);
    return SUCCESS;
  }
  std::lock_guard<std::mutex> lg(file_mutex_);
  json_file_.open(json_file_name_, std::ios::app);
  if (!json_file_.is_open()) {
    GELOGE(FAILED, "[Check][AnalyzeFile]analyze file does not exist[%s]",
           json_file_name_.c_str());
    REPORT_INNER_ERROR("E19999", "Analyze file %s dose not exist", json_file_name_.c_str());
    return PARAM_INVALID;
  }

  json jsn;
  GraphInfoToJson(jsn, *graph_info);
  bool ret_failed = false;
  try {
    json_file_ << jsn.dump(kJsonDumpLevel) << std::endl;
  } catch (nlohmann::detail::type_error &e) {
    GELOGE(FAILED,
           "[Json.dump][GraphInfo]Dump analyze file [%s] failed because [%s],"
           "session_id:%lu, graph_id:%lu",
           json_file_name_.c_str(), e.what(), session_id, graph_id);
    REPORT_INNER_ERROR("E19999", "Dump analyze file %s failed because %s, "
                       "session_id %lu, graph_id %lu",
                       json_file_name_.c_str(), e.what(), session_id, graph_id);
    ret_failed = true;
  }
  json_file_.close();
  return ret_failed ? FAILED : SUCCESS;
}

ge::Status Analyzer::DoAnalyze(DataInfo &data_info) {
  GELOGD("start to do analyzer process");

  auto pnode = data_info.node_ptr;
  GE_CHECK_NOTNULL(pnode);
  auto desc = pnode->GetOpDesc();
  GE_CHECK_NOTNULL(desc);
  // buff analyze data
  std::lock_guard<std::recursive_mutex> lg(mutex_);
  auto graph_info = GetJsonObject(data_info.session_id, data_info.graph_id);
  GE_CHECK_NOTNULL(graph_info);
  auto status = SaveOpInfo(desc, data_info, graph_info);
  if (status != SUCCESS) {
    GELOGE(status,
           "[Check][SaveOpInfo]save op info: desc_name [%s] desc_type [%s] failed!",
           desc->GetName().c_str(), desc->GetType().c_str());
    REPORT_CALL_ERROR("E19999", "Save op info: desc_name %s, desc_type %s failed",
                      desc->GetName().c_str(), desc->GetType().c_str());
    return FAILED;
  }
  // create json file
  return CreateAnalyzerFile();
}

ge::Status Analyzer::SaveOpInfo(ge::OpDescPtr desc, DataInfo &data_info,
                                std::shared_ptr<analyzer::GraphInfo> graph_info) {
  auto iter = errors_map.find(data_info.analyze_type);
  if (iter == errors_map.end()) {
    return PARAM_INVALID;
  }
  OpInfo op_info;
  op_info.error_type = iter->second;
  op_info.op_name = desc->GetName();
  op_info.op_type = desc->GetType();
  op_info.reason  = data_info.reason;

  for (const auto &ptr : desc->GetAllInputsDescPtr()) {
    TensorInfo tensor_info;
    tensor_info.shape  = ptr->GetShape().GetDims();
    tensor_info.d_type = ge::TypeUtils::DataTypeToSerialString(ptr->GetDataType());
    tensor_info.layout = ge::TypeUtils::FormatToSerialString(ptr->GetFormat());
    op_info.input_info.emplace_back(tensor_info);
  }
  for (const auto &ptr : desc->GetAllOutputsDescPtr()) {
    TensorInfo tensor_info;
    tensor_info.shape  = ptr->GetShape().GetDims();
    tensor_info.d_type = ge::TypeUtils::DataTypeToSerialString(ptr->GetDataType());
    tensor_info.layout = ge::TypeUtils::FormatToSerialString(ptr->GetFormat());
    op_info.output_info.emplace_back(tensor_info);
  }
  graph_info->op_info.emplace_back(op_info);

  return SUCCESS;
}

void Analyzer::TensorInfoToJson(json& j, const TensorInfo &tensor_info) {
  j[kShape] = tensor_info.shape;
  j[kDataType] = tensor_info.d_type;
  j[kLayout] = tensor_info.layout;
}

void Analyzer::OpInfoToJson(json& j, const OpInfo &op_info) {
  j[kErrorType] = op_info.error_type;
  j[kOpName] = op_info.op_name;
  j[kOpType] = op_info.op_type;
  j[kReason] = op_info.reason;
  for (size_t i = 0; i < op_info.input_info.size(); i++) {
    json json_tensor_info;
    TensorInfoToJson(json_tensor_info, op_info.input_info.at(i));
    j[kInput + std::to_string(i)] = json_tensor_info;
  }
  for (size_t i = 0; i < op_info.output_info.size(); i++) {
    json json_tensor_info;
    TensorInfoToJson(json_tensor_info, op_info.output_info.at(i));
    j[kOutput + std::to_string(i)] = json_tensor_info;
  }
}

void Analyzer::GraphInfoToJson(json& j, const GraphInfo &graph_info) {
  GELOGD("start to buff graph info!");
  j[kSessionId] = graph_info.session_id;
  j[kGraphId] = graph_info.graph_id;
  std::vector<json> json_op_infos;
  for (size_t i = 0; i < graph_info.op_info.size(); i++) {
    json json_op_info;
    OpInfoToJson(json_op_info, graph_info.op_info.at(i));
    json_op_infos.emplace_back(json_op_info);
  }
  j[kOp] = json_op_infos;
}
} // namespace ge
