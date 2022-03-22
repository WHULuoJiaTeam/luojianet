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

#include "graph/optimize/graph_optimize.h"

#include "graph/ge_context.h"
#include "common/local_context.h"
#include "graph/passes/dimension_adjust_pass.h"
#include "inc/pass_manager.h"
#include "init/gelib.h"

namespace {
const char *const kVectorCore = "VectorCore";
const char *const kVectorEngine = "VectorEngine";
const char *const kAicoreEngine = "AIcoreEngine";
const char *const kHostCpuEngine = "DNN_VM_HOST_CPU";
}  // namespace

namespace ge {
GraphOptimize::GraphOptimize()
    : optimize_type_(domi::FrameworkType::TENSORFLOW),
      cal_config_(""),
      insert_op_config_(""),
      core_type_("") {}

void AddNodeInputProperty(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return;
  }
  for (ge::NodePtr &node : compute_graph->GetDirectNode()) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return );
    auto in_control_anchor = node->GetInControlAnchor();
    vector<string> src_name_list;
    vector<string> input_name_list;
    vector<int64_t> src_index_list;
    GE_IF_BOOL_EXEC(
      in_control_anchor != nullptr, string src_name_temp; for (auto &out_control_anchor
                                                               : in_control_anchor->GetPeerOutControlAnchors()) {
        ge::NodePtr src_node = out_control_anchor->GetOwnerNode();
        GE_IF_BOOL_EXEC(src_node == nullptr, GELOGW("src_node is nullptr!"); continue);
        src_name_temp = src_name_temp == "" ? src_node->GetName() : src_name_temp + ":" + src_node->GetName();
      } GE_IF_BOOL_EXEC(src_name_temp != "", src_name_list.emplace_back(src_name_temp);
                        node_op_desc->SetSrcName(src_name_list);))

    for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
      auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

      ge::NodePtr src_node = peer_out_anchor->GetOwnerNode();
      src_index_list = node_op_desc->GetSrcIndex();
      src_name_list.emplace_back(src_node->GetName());
      src_index_list.emplace_back(peer_out_anchor->GetIdx());
      node_op_desc->SetSrcName(src_name_list);
      node_op_desc->SetSrcIndex(src_index_list);
      GE_IF_BOOL_EXEC(!(node_op_desc->GetType() == NETOUTPUT && GetLocalOmgContext().type == domi::TENSORFLOW),
                      ge::NodePtr peer_owner_node = peer_out_anchor->GetOwnerNode();
                      input_name_list.emplace_back(
                        peer_owner_node->GetName() +
                        (peer_out_anchor->GetIdx() == 0 ? "" : ": " + to_string(peer_out_anchor->GetIdx())));
                      node_op_desc->SetInputName(input_name_list);)
    }
  }
}

Status GraphOptimize::OptimizeSubGraph(ComputeGraphPtr &compute_graph, const std::string &engine_name) {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  Status ret = SUCCESS;
  vector<GraphOptimizerPtr> graph_optimizer;

  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid, graph:%s",
                       compute_graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib not init before, graph:%s",
           compute_graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  if (instance_ptr->DNNEngineManagerObj().IsEngineRegistered(engine_name)) {
    instance_ptr->OpsKernelManagerObj().GetGraphOptimizerByEngine(engine_name, graph_optimizer);
    AddNodeInputProperty(compute_graph);

    if (compute_graph->GetDirectNode().size() == 0) {
      GELOGW("[OptimizeSubGraph] compute_graph do not has any node.");
      return SUCCESS;
    }

    if (build_mode_ == BUILD_MODE_TUNING && (build_step_ == BUILD_STEP_AFTER_UB_MATCH
      || build_step_ == BUILD_STEP_AFTER_MERGE)) {
      for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
        Status ret = (*iter)->OptimizeFusedGraphAfterGraphSlice(*(compute_graph));
        if (ret != SUCCESS) {
          REPORT_INNER_ERROR("E19999", "Call OptimizeFusedGraphAfterGraphSlice failed, ret:%d, engine_name:%s, "
                             "graph_name:%s", ret, engine_name.c_str(),
                             compute_graph->GetName().c_str());
          GELOGE(ret, "[Call][OptimizeFusedGraphAfterGraphSlice] failed, ret:%d, engine_name:%s, graph_name:%s",
                 ret, engine_name.c_str(), compute_graph->GetName().c_str());
          return ret;
        }
      }
      return SUCCESS;
    }

    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      ret = (*iter)->OptimizeFusedGraph(*(compute_graph));
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Call OptimizeFusedGraph failed, ret:%d, engine_name:%s, "
                           "graph_name:%s", ret, engine_name.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Optimize][FusedGraph] failed, ret:%d, engine_name:%s, graph_name:%s",
               ret, engine_name.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  } else {
    GELOGI("Engine: %s is not registered. do nothing in subGraph Optimize by ATC.", engine_name.c_str());
  }

  return ret;
}

Status GraphOptimize::OptimizeOriginalGraph(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  Status ret = SUCCESS;
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid, graph:%s.",
                       compute_graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib not init before, graph:%s.",
           compute_graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGI("optimize by opskernel in original graph optimize phase. num of graph_optimizer is %zu.",
         graph_optimizer.size());
  string exclude_core_Type = (core_type_ == kVectorCore) ? kAicoreEngine : kVectorEngine;
  GELOGD("[OptimizeOriginalGraph]: engine type will exclude: %s", exclude_core_Type.c_str());
  if (graph_optimizer.size() != 0) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (iter->first == exclude_core_Type) {
        continue;
      }
      if (GetContext().GetHostExecFlag() && iter->first != kHostCpuEngine) {
        // graph exec on host, no need OptimizeOriginalGraph for other engine.
        continue;
      }
      ret = (iter->second)->OptimizeOriginalGraph(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Call OptimizeOriginalGraph failed, ret:%d, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Optimize][OriginalGraph] failed, ret:%d, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeOriginalGraphJudgeInsert(ComputeGraphPtr &compute_graph) {
  GELOGD("OptimizeOriginalGraphJudgeInsert in");

  GE_CHECK_NOTNULL(compute_graph);
  Status ret = SUCCESS;
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid, graph:%s",
                       compute_graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib not init before, graph:%s",
           compute_graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGI("optimize by opskernel in judging insert phase. num of graph_optimizer is %zu.",
         graph_optimizer.size());
  string exclude_core_Type = (core_type_ == kVectorCore) ? kAicoreEngine : kVectorEngine;
  if (graph_optimizer.size() != 0) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (iter->first == exclude_core_Type) {
        GELOGI("[OptimizeOriginalGraphJudgeInsert]: engine type will exclude: %s", exclude_core_Type.c_str());
        continue;
      }
      if (GetContext().GetHostExecFlag() && iter->first != kHostCpuEngine) {
        // graph exec on host, no need OptimizeOriginalGraphJudgeInsert for other engine.
        continue;
      }
      GELOGI("Begin to refine running format by engine %s", iter->first.c_str());
      ret = (iter->second)->OptimizeOriginalGraphJudgeInsert(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Call OptimizeOriginalGraphJudgeInsert failed, ret:%d, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeOriginalGraphJudgeInsert] failed, ret:%d, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeOriginalGraphForQuantize(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid, graph:%s.",
                       compute_graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][Gelib] Gelib not init before, graph:%s.",
           compute_graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGI("optimize by opskernel in original graph optimize quantize phase. num of graph_optimizer is %zu.",
         graph_optimizer.size());
  Status ret = SUCCESS;
  string exclude_core_Type = (core_type_ == kVectorCore) ? kAicoreEngine : kVectorEngine;
  GELOGD("[OptimizeOriginalGraphForQuantize]: engine type will exclude: %s", exclude_core_Type.c_str());
  if (graph_optimizer.size() != 0) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (iter->first == exclude_core_Type || iter->second == nullptr) {
        continue;
      }
      ret = iter->second->OptimizeGraphPrepare(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Call OptimizeGraphPrepare failed, ret:%d, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeGraphPrepare] failed, ret:%d, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeGraphBeforeBuildForRts(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid, graph:%s.",
                       compute_graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib not init before, graph:%s.",
           compute_graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGD("optimize by opskernel in graph optimize before build phase. num of graph_optimizer is %zu.",
         graph_optimizer.size());
  Status ret = SUCCESS;
  string exclude_core_Type = (core_type_ == kVectorCore) ? kAicoreEngine : kVectorEngine;
  GELOGD("[OptimizeGraphBeforeBuildForRts]: engine type will exclude: %s, core_type_: %s",
         exclude_core_Type.c_str(), core_type_.c_str());
  if (graph_optimizer.size() != 0) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (iter->first == exclude_core_Type || iter->second == nullptr) {
        continue;
      }
      ret = iter->second->OptimizeGraphBeforeBuild(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Call OptimizeGraphBeforeBuild failed, ret:%d, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeGraphBeforeBuild] failed, ret:%d, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeAfterStage1(ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);
  GELOGD("OptimizeAfterStage1 in");
  if (GetContext().GetHostExecFlag()) {
    // graph exec on host, no need OptimizeAfterStage1
    return SUCCESS;
  }

  Status ret = SUCCESS;
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "OptimizeAfterStage1 failed.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGI("Optimize by ops kernel in after stage1 phase, num of graph_optimizer is %zu.", graph_optimizer.size());
  string exclude_core_type = (core_type_ == kVectorCore) ? kAicoreEngine : kVectorEngine;
  if (graph_optimizer.size() != 0) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (iter->first == exclude_core_type) {
        GELOGI("[OptimizeAfterStage1]: engine type will exclude:%s.", exclude_core_type.c_str());
        continue;
      }
      GELOGI("Begin to optimize graph after stage1 by engine %s.", iter->first.c_str());
      ret = (iter->second)->OptimizeAfterStage1(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Call OptimizeAfterStage1 failed, ret:%d, engine_name:%s, "
                           "graph_name:%s.", ret, iter->first.c_str(), compute_graph->GetName().c_str());
        GELOGE(ret, "[OptimizeAfterStage1]: graph optimize failed, ret:%d.", ret);
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::SetOptions(const ge::GraphManagerOptions &options) {
  if (options.framework_type >= static_cast<int32_t>(domi::FrameworkType::FRAMEWORK_RESERVED)) {
    REPORT_INNER_ERROR("E19999", "Param framework_type:%d in option check invalid",
                       options.framework_type);
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Optimize Type %d invalid.", options.framework_type);
    return GE_GRAPH_OPTIONS_INVALID;
  }
  optimize_type_ = static_cast<domi::FrameworkType>(options.framework_type);
  cal_config_ = options.calibration_conf_file;
  insert_op_config_ = options.insert_op_file;
  train_graph_flag_ = options.train_graph_flag;
  local_fmk_op_flag_ = options.local_fmk_op_flag;
  func_bin_path_ = options.func_bin_path;
  core_type_ = options.core_type;
  build_mode_ = options.build_mode;
  build_step_ = options.build_step;
  return SUCCESS;
}

void GraphOptimize::TranFrameOp(ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL_JUST_RETURN(compute_graph);
  vector<string> local_framework_op_vec = {
    "TensorDataset", "QueueDataset", "DeviceQueueDataset", "ParallelMapDataset", "BatchDatasetV2",
    "IteratorV2",    "MakeIterator", "IteratorGetNext",    "FilterDataset",      "MapAndBatchDatasetV2"};
  for (auto &nodePtr : compute_graph->GetAllNodes()) {
    OpDescPtr op = nodePtr->GetOpDesc();
    GE_IF_BOOL_EXEC(op == nullptr, GELOGW("op is nullptr!"); continue);
    // fwkop black-white sheet
    vector<string>::iterator iter =
      std::find(local_framework_op_vec.begin(), local_framework_op_vec.end(), op->GetType());
    if (iter != local_framework_op_vec.end()) {
      // set - original_type
      if (!AttrUtils::SetStr(op, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, op->GetType())) {
        GELOGW("TranFrameOp SetStr ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE failed");
      }
      // set - framework_type
      // [No need to verify return value]
      op->SetType("FrameworkOp");
      if (!AttrUtils::SetInt(op, ATTR_NAME_FRAMEWORK_FWK_TYPE, domi::FrameworkType::TENSORFLOW)) {
        GELOGW("TranFrameOp SetInt ATTR_NAME_FRAMEWORK_FWK_TYPE failed");
      }
    }
  }
}

Status GraphOptimize::IdentifyReference(ComputeGraphPtr &compute_graph) {
  for (auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto input_name_index = op_desc->GetAllInputName();
    bool is_ref = false;
    for (const auto &name_index : input_name_index) {
      const int out_index = op_desc->GetOutputIndexByName(name_index.first);
      if (out_index != -1) {
        auto input_desc = op_desc->GetInputDesc(name_index.second);
        input_desc.SetRefPortByIndex({name_index.second});
        op_desc->UpdateInputDesc(name_index.second, input_desc);
        GELOGI("SetRefPort: set op[%s] input desc[%u-%s] ref.",
               op_desc->GetName().c_str(), name_index.second, name_index.first.c_str());
        auto output_desc = op_desc->GetOutputDesc(static_cast<uint32_t>(out_index));
        output_desc.SetRefPortByIndex({name_index.second});
        op_desc->UpdateOutputDesc(static_cast<uint32_t>(out_index), output_desc);
        GELOGI("SetRefPort: set op[%s] output desc[%u-%s] ref.",
               op_desc->GetName().c_str(), out_index, name_index.first.c_str());
        is_ref = true;
      }
    }
    if (is_ref) {
      AttrUtils::SetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
      GELOGI("param [node] %s is reference node, set attribute %s to be true.",
             node->GetName().c_str(), ATTR_NAME_REFERENCE.c_str());
    }
  }
  return SUCCESS;
}
Status GraphOptimize::OptimizeWholeGraph(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid, graph:%s.",
                       compute_graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib not init before, graph:%s.",
           compute_graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGI("optimize by opskernel in OptimizeWholeGraph. num of graph_optimizer is %zu.", graph_optimizer.size());
  Status ret = SUCCESS;
  string exclude_core_type = (core_type_ == kVectorCore) ? kAicoreEngine : kVectorEngine;
  GELOGD("[OptimizeWholeGraph]: engine type will exclude: %s", exclude_core_type.c_str());
  if (!graph_optimizer.empty()) {
    for (auto &iter : graph_optimizer) {
      if (iter.first == exclude_core_type || iter.second == nullptr) {
        continue;
      }
      GELOGI("Begin to optimize whole graph by engine %s", iter.first.c_str());
      ret = iter.second->OptimizeWholeGraph(*compute_graph);
      GE_DUMP(compute_graph, "OptimizeWholeGraph" + iter.first);
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "Call OptimizeWholeGraph failed, ret:%d, engine_name:%s, "
                           "graph_name:%s", ret, iter.first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeWholeGraph] failed, ret:%d, engine_name:%s, graph_name:%s",
               ret, iter.first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}
}  // namespace ge
