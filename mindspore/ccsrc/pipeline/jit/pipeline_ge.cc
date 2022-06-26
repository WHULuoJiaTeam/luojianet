/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/pipeline_ge.h"

#include <sstream>
#include <map>
#include <cstdlib>
#include <algorithm>

#include "utils/hash_map.h"
#include "include/common/debug/anf_ir_dump.h"
#include "ir/tensor.h"
#include "include/transform/graph_ir/convert.h"
#include "include/transform/graph_ir/df_graph_manager.h"
#include "include/transform/graph_ir/graph_builder.h"
#include "include/transform/graph_ir/graph_runner.h"
#include "include/common/debug/draw.h"
#include "abstract/abstract_value.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;
using TensorOrderMap = std::map<std::string, std::shared_ptr<Tensor>>;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using mindspore::transform::DfGraphConvertor;
using mindspore::transform::DfGraphManager;
using mindspore::transform::GeTensorPtr;
using mindspore::transform::MeTensorPtr;
using mindspore::transform::Status;
using mindspore::transform::TransformUtil;

void DoExecNonInputGraph(const std::string &phase) {
  std::vector<GeTensorPtr> ge_tensors;
  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;
  run_options.name = phase;
  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return;
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    if (ret != Status::SUCCESS) {
      MS_LOG(ERROR) << "Exec graph:" << run_options.name << " failed";
      return;
    }
  }
}

void SetGeOption(const std::map<std::string, std::string> &options) {
  ConfigManager::GetInstance().set_ge_initialize_options(options);
}

Status CreateSessionAndGraphRunner(bool is_training = true) {
  std::shared_ptr<ge::Session> sess = DfGraphManager::GetInstance().GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options;
    if (is_training) {
      options["ge.trainFlag"] = "1";
      options["ge.streamNum"] = "100";
      options["ge.enabledLocalFmkop"] = "1";
      options["ge.hcomParallel"] = "1";
    } else {
      options["ge.trainFlag"] = "0";
    }

    options["ge.enablePrintOpPass"] = "0";
    sess = transform::GraphRunner::NewSession(options);
    DfGraphManager::GetInstance().SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = std::make_shared<transform::GraphRunner>(options);
  DfGraphManager::GetInstance().SetGraphRunner(graph_runner);
  return Status::SUCCESS;
}

bool InitExecDatasetGe(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, const std::string &phase) {
  std::vector<int64_t> ge_types;
  (void)std::transform(types.begin(), types.end(), std::back_inserter(ge_types), [](const TypePtr &i) -> int64_t {
    return transform::TransformUtil::ConvertDataType(i->type_id());
  });

  ConfigManager::GetInstance().set_dataset_mode(DatasetMode::DS_SINK_MODE);
  ConfigManager::GetInstance().set_iter_num(queue_name, size);
  ConfigManager::GetInstance().set_dataset_phase(phase);

  DatasetGraphParam param(queue_name, size, batch_size, ge_types, shapes, input_indexes);
  ConfigManager::GetInstance().set_dataset_param(param);

  if (transform::BuildDatasetGraph(param, phase) != transform::SUCCESS) {
    MS_LOG(ERROR) << "Build dateset graph failed.";
    return false;
  }

  auto env_ge = common::GetEnv("MS_ENABLE_GE");
  auto env_training = common::GetEnv("MS_GE_TRAIN");
  bool training = false;
  if (env_ge == "1" && env_training == "1") {
    training = true;
  }
  if (training) {
    (void)setenv("GE_TRAIN", "1", 1);
  } else {
    (void)setenv("GE_TRAIN", "0", 1);
  }

  if (CreateSessionAndGraphRunner(training) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return false;
  }

  MS_LOG(INFO) << "DoExecNonInputGraph:" << phase;
  DoExecNonInputGraph(phase);

  return true;
}

void ConvertObjectToTensors(const py::dict &dict, TensorOrderMap *const tensors) {
  for (auto item : dict) {
    if ((!py::isinstance<py::str>(item.first))) {
      MS_LOG(WARNING) << "Type of key of py_dict is not string, ignore it.";
      continue;
    }
    std::shared_ptr<Tensor> tensor;
    std::string name = py::cast<std::string>(item.first);
    if (py::isinstance<py::float_>(item.second.attr("data"))) {
      // convert float to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeFloat32, std::vector<int64_t>({1}));
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (py::isinstance<py::int_>(item.second.attr("data"))) {
      // convert int64_t to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeInt32, std::vector<int64_t>({1}));
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (py::isinstance<Tensor>(item.second.attr("data"))) {
      // cast tensor
      tensor = py::cast<std::shared_ptr<Tensor>>(item.second.attr("data"));
    }

    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Get default value for " << name << " failed";
    }
    (void)tensors->emplace(name, tensor);
  }
}

bool AddDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::dict &init_params,
                const std::string &phase, const py::object &broadcast_params) {
  FuncGraphPtr anf_graph = info.at(phase)->func_graph;
  DfGraphConvertor converter(anf_graph);

  size_t pos = phase.find('.');
  std::string net_id = ((pos == std::string::npos || pos == phase.size() - 1) ? phase : phase.substr(pos + 1));
  std::string phase_prefix = phase.substr(0, pos);
  if (phase_prefix == "export") {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    converter.set_training(false);
  }

  TensorOrderMap init_tensors{};
  ConvertObjectToTensors(init_params, &init_tensors);
  (void)converter.ConvertAllNode().InitParam(init_tensors).BuildGraph();

  if (!broadcast_params.is_none()) {
    if (!py::isinstance<py::dict>(broadcast_params)) {
      MS_LOG(ERROR) << "Invalid broadcast params, it must be py::dict type";
      return false;
    }
    py::dict broadcast = broadcast_params.cast<py::dict>();
    if (broadcast.empty()) {
      (void)converter.GenerateBroadcastGraph(init_tensors);
    } else {
      TensorOrderMap broadcast_tensors{};
      ConvertObjectToTensors(broadcast, &broadcast_tensors);
      (void)converter.GenerateBroadcastGraph(broadcast_tensors);
    }
    MS_LOG(INFO) << "Generate broadcast graph with params and broadcast_empty is " << broadcast.empty();
  }

  (void)converter.GenerateCheckpointGraph();
  if (converter.ErrCode() != 0) {
    DfGraphManager::GetInstance().ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << converter.ErrCode();
    return false;
  }
#ifdef ENABLE_DUMP_IR
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    converter.DrawComputeGraph(GetSaveGraphsPathName("ge_graph.dot"));                      // for debug
    converter.DrawInitGraph(GetSaveGraphsPathName("init_graph.dot"));                       // for debug
    converter.DrawSaveCheckpointGraph(GetSaveGraphsPathName("save_checkpoint_graph.dot"));  // for debug
  }
#endif
  std::string init_graph = "init_subgraph." + net_id;
  std::string checkpoint_name = "save." + net_id;
  if (phase.find("train") != std::string::npos) {
    (void)DfGraphManager::GetInstance().AddGraph(phase, converter.GetComputeGraph(), {{"ge.exec.variable_acc", "1"}});
  } else {
    (void)DfGraphManager::GetInstance().AddGraph(phase, converter.GetComputeGraph());
  }
  (void)DfGraphManager::GetInstance().AddGraph(init_graph, converter.GetInitGraph());
  (void)DfGraphManager::GetInstance().AddGraph(BROADCAST_GRAPH_NAME, converter.GetBroadcastGraph());

  Status ret = DfGraphManager::GetInstance().AddGraph(checkpoint_name, converter.GetSaveCheckpointGraph());
  if (ret == Status::SUCCESS) {
    DfGraphManager::GetInstance().SetAnfGraph(checkpoint_name, anf_graph);
  }

  return true;
}

FuncGraphPtr BuildDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::dict &init_params,
                          const std::string &phase, const py::object &broadcast_params) {
  if (info.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << GetPhasePrefix(phase);
  }
  FuncGraphPtr anf_graph = info.at(phase)->func_graph;
#ifdef ENABLE_DUMP_IR
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    draw::Draw("anf_graph.dot", anf_graph);  // for debug
    DumpIR("anf_graph.ir", anf_graph, true);
  }
#endif

  if (!AddDFGraph(info, init_params, phase, broadcast_params)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

  auto env_ge = common::GetEnv("MS_ENABLE_GE");
  auto env_training = common::GetEnv("MS_GE_TRAIN");
  bool training = false;
  if (env_ge == "1" && env_training == "1") {
    training = true;
  }
  if (training) {
    (void)setenv("GE_TRAIN", "1", 1);
  } else {
    (void)setenv("GE_TRAIN", "0", 1);
  }

  if (CreateSessionAndGraphRunner(training) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return nullptr;
  }

  return anf_graph;
}

void RunGEInitGraph(const py::dict &init_params, const std::string &phase) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  TensorOrderMap inputs_with_name{};
  ConvertObjectToTensors(init_params, &inputs_with_name);
  std::vector<tensor::TensorPtr> inputs;
  (void)std::transform(inputs_with_name.begin(), inputs_with_name.end(), std::back_inserter(inputs),
                       [](const std::pair<std::string, tensor::TensorPtr> &item) { return item.second; });

  std::vector<GeTensorPtr> ge_tensors = TransformUtil::ConvertInputTensors(inputs, kOpFormat_NCHW);
  if (ge_tensors.size() != inputs.size()) {
    MS_LOG(ERROR) << "Args convert to ge tensor error.";
    return;
  }
  MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size() << ".";

  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;

  run_options.name = phase;
  if (DfGraphManager::GetInstance().GetGraphByName(phase) == nullptr) {
    MS_LOG(WARNING) << "Can not find " << phase << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    if (ret != Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << phase << " graph failed.";
    }

    MS_LOG(INFO) << "Exec " << phase << " graph success.";

    if ((ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION) &&
        (DfGraphManager::GetInstance().GetGraphByName(BROADCAST_GRAPH_NAME) != nullptr)) {
      run_options.name = BROADCAST_GRAPH_NAME;
      ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
      if (ret != Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec BROADCAST_GRAPH_NAME failed.";
      }
      MS_LOG(INFO) << "Exec broadcast graph success.";
    }
  }
}

py::object ExtractGeneralCnodeRet(const AbstractBasePtr &cnode_data, const py::tuple &data, size_t *count) {
  MS_EXCEPTION_IF_NULL(cnode_data);

  if (cnode_data->isa<AbstractTensor>()) {
    if (*count >= data.size()) {
      MS_LOG(EXCEPTION) << "The number of elements in the outputs : " << data.size()
                        << " less than the number of elements required. ";
    }

    BaseShapePtr shape = cnode_data->BuildShape();
    if (!shape->isa<abstract::Shape>()) {
      MS_LOG(EXCEPTION) << "The shape of the tensor derived is not Shape, is " << shape->ToString();
    }

    auto shape_me = shape->cast<abstract::ShapePtr>()->shape();
    auto shape_ge = py::cast<Tensor &>(data[*count]).shape();
    if (shape_ge != shape_me) {  // dynamic shape
      MS_LOG(WARNING) << "The shape of the " << *count << "th tensor returned: " << shape_ge
                      << " is not the same as the shape of the tensor derived: " << shape_me;
    }

    return data[(*count)++];
  }

  if (!cnode_data->isa<AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "The output of operator in the final anf graph could "
                      << "only be a tensor or a tuple of tensor, but got " << cnode_data->BuildValue()->ToString()
                      << ".";
  }
  auto data_tp = cnode_data->cast<AbstractTuplePtr>();
  auto elements = data_tp->elements();
  size_t size = data_tp->size();
  auto tp = py::tuple(size);
  for (size_t i = 0; i < size; i++) {
    tp[i] = ExtractGeneralCnodeRet(elements[i], data, count);
  }
  return std::move(tp);
}

py::object StructureOutput(const AnfNodePtr &output_node, const py::tuple &data, size_t *count) {
  MS_EXCEPTION_IF_NULL(output_node);

  if (output_node->isa<ValueNode>()) {
    return ValueToPyData(GetValueNode(output_node));
  }

  if (output_node->isa<Parameter>()) {
    if (*count >= data.size()) {
      MS_LOG(EXCEPTION) << "The number of elements in the outputs : " << data.size()
                        << " less than the number of elements required. ";
    }
    return data[(*count)++];
  }

  auto output_c = output_node->cast<CNodePtr>();
  if (output_c == nullptr) {
    MS_LOG(EXCEPTION) << "The final anf graph could only have constant, parameter, and operator, but got "
                      << output_node->ToString();
  }

  if (output_c->IsApply(prim::kPrimMakeTuple)) {
    auto input_list = output_c->inputs();
    size_t size = input_list.size();
    auto tp = py::tuple(size - 1);
    for (size_t i = 1; i < size; i++) {
      tp[i - 1] = StructureOutput(input_list[i], data, count);
    }
    return std::move(tp);
  }
  if (output_c->IsApply(prim::kPrimDepend)) {
    return StructureOutput(output_c->input(1), data, count);
  }

  return ExtractGeneralCnodeRet(output_c->abstract(), data, count);
}

std::shared_ptr<py::object> DoExecGraph(const FuncGraphPtr &graph, const std::vector<MeTensorPtr> &inputs,
                                        const std::string &phase) {
  std::vector<GeTensorPtr> ge_tensors = TransformUtil::ConvertInputTensors(inputs, kOpFormat_NCHW);
  if (ge_tensors.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Convert me args to ge tensor error.";
  }

  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;
  run_options.name = phase;
  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
    if (ret != Status::SUCCESS) {
      MS_LOG(ERROR) << "Exec graph failed";
      return nullptr;
    }
  }

  std::vector<MeTensorPtr> me_outputs = TransformUtil::ConvertGeTensors(ge_outputs);
  if (me_outputs.size() != ge_outputs.size()) {
    MS_LOG(WARNING) << "Convert output Ge tensor to Me tensor failed";
  }

  py::tuple outputs(me_outputs.size());
  for (std::size_t i = 0; i < outputs.size(); i++) {
    outputs[i] = *me_outputs[i];
  }

  std::shared_ptr<py::object> ret = nullptr;

  AnfNodePtr output_node = graph->get_return()->input(1);
  MS_EXCEPTION_IF_NULL(output_node);
  size_t count = 0;
  py::object oj = StructureOutput(output_node, outputs, &count);
  ret = std::make_shared<py::object>(oj);

  return ret;
}

void ProcessGeArg(const std::map<std::string, ExecutorInfoPtr> &info, const py::tuple &args, const std::string &phase,
                  std::vector<tensor::TensorPtr> *inputs) {
  // check the arg and use the GraphExecutorPy args
  std::size_t size = args.size();

  if (info.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << GetPhasePrefix(phase);
  }

  auto arg_size = info.at(phase)->arg_list_size;
  if (size != arg_size) {
    MS_LOG(EXCEPTION) << "The real arg num : size = " << size << ". graph_arg_size = " << arg_size;
  }

  // process the first args of tensor
  // only in dataset normal(non-sink) mode, fp_bp graph need input tensors
  if (ConfigManager::GetInstance().dataset_mode() == DS_NORMAL_MODE) {
    for (std::size_t i = 0; i < size; i++) {
      ValuePtr converted = nullptr;
      bool succ = parse::ConvertData(args[i], &converted);
      if (!succ) {
        MS_LOG(EXCEPTION) << "The " << i << "th arg convert failed.";
      }
      if (converted->isa<tensor::Tensor>()) {
        inputs->push_back(converted->cast<tensor::TensorPtr>());
      } else {
        MS_EXCEPTION(TypeError) << "The " << i << "th arg: " << converted->ToString() << " is not tensor.";
      }
    }
  }
}

py::object ExecDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::tuple &args,
                       const std::string &phase) {
  std::string phase_prefix = GetPhasePrefix(phase);
  if (phase_prefix == "save") {
    DoExecNonInputGraph(phase);
    ConfigManager::GetInstance().ResetConfig();
    return py::none();
  }

  if (info.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "There is no phase:" << phase;
  }
  FuncGraphPtr anf_graph = info.at(phase)->func_graph;

  std::shared_ptr<py::object> ret_val = std::make_shared<py::object>();
  // We will not execute graph when output is constant or just input itself.
  if (IsGraphOutputValueNodeOrParameter(info.at(phase)->func_graph->output(), args, ret_val)) {
    ConfigManager::GetInstance().ResetConfig();
    return *ret_val;
  }

  std::vector<tensor::TensorPtr> inputs;
  ProcessGeArg(info, args, phase, &inputs);

  std::shared_ptr<py::object> ret = DoExecGraph(anf_graph, inputs, phase);
  ConfigManager::GetInstance().ResetConfig();
  if (ret != nullptr) {
    return *ret;
  } else {
    MS_LOG(EXCEPTION) << "Exec graph failed";
  }
}

void ExportDFGraph(const std::string &file_name, const std::string &phase) {
  MS_LOG(DEBUG) << "Export graph begin.";
  transform::DfGraphWrapperPtr wrap_ptr = DfGraphManager::GetInstance().GetGraphByName(phase);
  if (wrap_ptr == nullptr) {
    MS_LOG(ERROR) << "Get graph form DfGraphManager failed!";
    return;
  }

  transform::DfGraphPtr ge_graph = wrap_ptr->graph_ptr_;
  if (ge_graph == nullptr) {
    MS_LOG(ERROR) << "Graph is null!";
    return;
  }

  if (ge_graph->SaveToFile(file_name) != 0) {
    MS_LOG(EXCEPTION) << "Export air model failed.";
  }
  MS_LOG(INFO) << "Export air model finish.";
}
}  // namespace pipeline
}  // namespace mindspore
