/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "cxx_api/model/acl/model_converter.h"
#include <memory>
#include "include/transform/graph_ir/convert.h"
#include "include/transform/graph_ir/graph_runner.h"
#include "mindspore/core/utils/ms_context.h"
#include "include/api/serialization.h"
#include "graph/model.h"
#include "cxx_api/model/model_converter_utils/multi_process.h"

namespace mindspore {
namespace {
transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

bool CreateSessionAndGraphRunner() {
  std::shared_ptr<ge::Session> sess = transform::DfGraphManager::GetInstance().GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options;
    options["ge.trainFlag"] = "0";
    options["ge.enablePrintOpPass"] = "0";
    sess = transform::GraphRunner::NewSession(options);
    transform::DfGraphManager::GetInstance().SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = std::make_shared<transform::GraphRunner>(options);
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Create new graph runner failed";
    return false;
  } else {
    transform::DfGraphManager::GetInstance().SetGraphRunner(graph_runner);
  }

  return true;
}
}  // namespace

transform::DfGraphPtr ModelConverter::ConvertFuncGraphToAIR(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::DfGraphConvertor converter(anf_graph);
  std::string net_id = "0";
  std::string init_graph = "init_subgraph." + net_id;
  std::string checkpoint_name = "save." + net_id;

  converter.set_training(false);
  (void)converter.ConvertAllNode().InitParam(GetParams(anf_graph)).BuildGraph();
  (void)converter.GenerateCheckpointGraph();
  if (converter.ErrCode() != 0) {
    transform::DfGraphManager::GetInstance().ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << converter.ErrCode();
    return nullptr;
  }
  (void)transform::DfGraphManager::GetInstance().AddGraph(anf_graph->ToString(), converter.GetComputeGraph());
  (void)transform::DfGraphManager::GetInstance().AddGraph(init_graph, converter.GetInitGraph());
  (void)transform::DfGraphManager::GetInstance().AddGraph(BROADCAST_GRAPH_NAME, converter.GetBroadcastGraph());

  transform::Status ret =
    transform::DfGraphManager::GetInstance().AddGraph(checkpoint_name, converter.GetSaveCheckpointGraph());
  if (ret == transform::Status::SUCCESS) {
    transform::DfGraphManager::GetInstance().SetAnfGraph(checkpoint_name, anf_graph);
  }

  (void)setenv("GE_TRAIN", "0", 1);

  if (!CreateSessionAndGraphRunner()) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return nullptr;
  }

  auto wrap_ptr = transform::DfGraphManager::GetInstance().GetGraphByName(anf_graph->ToString());
  if (wrap_ptr == nullptr) {
    MS_LOG(ERROR) << "Get graph form DfGraphManager failed!";
    return nullptr;
  }
  transform::DfGraphPtr &ge_graph = wrap_ptr->graph_ptr_;
  if (ge_graph == nullptr) {
    MS_LOG(ERROR) << "The export graph is null";
    return nullptr;
  }

  return ge_graph;
}

Buffer ModelConverter::BuildAirModel(const transform::DfGraphPtr &graph,
                                     const std::map<std::string, std::string> &init_options,
                                     const std::map<std::string, std::string> &build_options) {
  ge::ModelBufferData model;
  auto ret = ge::aclgrphBuildInitialize(init_options);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphBuildInitialize fail.";
    return Buffer();
  }

  ret = ge::aclgrphBuildModel(*graph, build_options, model);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphBuildModel fail.";
    return Buffer();
  }

  if (SaveModel(model) != kSuccess) {
    MS_LOG(ERROR) << "Save model failed.";
    return Buffer();
  }

  ge::aclgrphBuildFinalize();
  return Buffer(model.data.get(), model.length);
}

Status ModelConverter::SaveModel(const ge::ModelBufferData &model) {
#ifdef BUILD_LITE
  std::string file_path;
  auto option = options_.lock();
  if (option != nullptr) {
    file_path = option->GetOmFilePath();
  }
  if (file_path.empty()) {
    MS_LOG(INFO) << "File path is empty, there is no need to save model";
    return kSuccess;
  }
  MS_LOG(INFO) << "Om file path: " << file_path;
  auto ret = ge::aclgrphSaveModel(file_path, model);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphSaveModel fail.";
    return kMCFailed;
  }
#endif
  return kSuccess;
}

Buffer ModelConverter::LoadMindIR(const FuncGraphPtr &func_graph) {
  MultiProcess multi_process;
  Buffer buffer_ret;
  auto parent_process = [&func_graph, &buffer_ret, this](MultiProcess *multi_process) -> Status {
    MS_EXCEPTION_IF_NULL(multi_process);
    auto df_graph = ConvertFuncGraphToAIR(func_graph);
    if (df_graph == nullptr) {
      MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
      return kMCFailed;
    }
    ge::Model model;
    ge::Buffer model_data;
    model.SetGraph(*df_graph);
    auto ge_ret = model.Save(model_data);
    if (ge_ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Save ge model to buffer failed.";
      return kMCFailed;
    }

    // send original model to child
    auto status = multi_process->SendMsg(model_data.data(), model_data.size());
    if (status != kSuccess) {
      MS_LOG_ERROR << "Send original model to child process failed";
      return status;
    }
    // receive convert model result from child
    CreateBufferCall call = [&buffer_ret](size_t msg_len) -> uint8_t * {
      (void)buffer_ret.ResizeData(msg_len);
      return reinterpret_cast<uint8_t *>(buffer_ret.MutableData());
    };
    status = multi_process->ReceiveMsg(call);
    if (status != kSuccess) {
      MS_LOG_ERROR << "Receive result model from child process failed";
      return status;
    }
    return kSuccess;
  };
  auto child_process = [this](MultiProcess *multi_process) -> Status {
    MS_EXCEPTION_IF_NULL(multi_process);
    // receive original model from parent
    Buffer model;
    CreateBufferCall call = [&model](size_t msg_len) -> uint8_t * {
      (void)model.ResizeData(msg_len);
      return reinterpret_cast<uint8_t *>(model.MutableData());
    };
    auto status = multi_process->ReceiveMsg(call);
    if (status != kSuccess) {
      MS_LOG_ERROR << "Receive original model from parent process failed";
      return status;
    }
    Buffer model_result = LoadAscendIRInner(model);
    if (model_result.DataSize() == 0) {
      MS_LOG_ERROR << "Convert model from MindIR to OM failed";
      return kMCFailed;
    }
    // send result model to parent
    status = multi_process->SendMsg(model_result.Data(), model_result.DataSize());
    if (status != kSuccess) {
      MS_LOG_ERROR << "Send result model to parent process failed";
      return status;
    }
    return kSuccess;
  };
  auto status = multi_process.MainProcess(parent_process, child_process);
  if (status != kSuccess) {
    MS_LOG_ERROR << "Convert MindIR model to OM model failed";
  } else {
    MS_LOG_INFO << "Convert MindIR model to OM model success";
  }
  return buffer_ret;
}

Buffer ModelConverter::LoadAscendIRInner(const Buffer &model_data) {
  ge::Model load_model = ge::Model("loadmodel", "version2");
  ge::Status ret =
    ge::Model::Load(reinterpret_cast<const uint8_t *>(model_data.Data()), model_data.DataSize(), load_model);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Load AscendIR failed, ret = " << ret;
    return Buffer();
  }

  transform::DfGraphPtr df_graph = std::make_shared<transform::DfGraph>(load_model.GetGraph());
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return Buffer();
  }

  std::map<std::string, std::string> init_options;
  std::map<std::string, std::string> build_options;
  auto option = options_.lock();
  if (option != nullptr) {
    std::tie(init_options, build_options) = option->GenAclOptions();
  }

  return BuildAirModel(df_graph, init_options, build_options);
}
}  // namespace mindspore
