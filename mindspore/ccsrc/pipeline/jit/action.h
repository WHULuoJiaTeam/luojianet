/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_ACTION_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_ACTION_H_

#include <vector>
#include <functional>
#include <utility>
#include <string>
#include "pipeline/jit/resource.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/graph_compiler/backend.h"

namespace mindspore {
extern const char kMsConvert[];

namespace pipeline {
using ActionItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;

bool ParseAction(const ResourcePtr &res);
bool SymbolResolveAction(const ResourcePtr &res);
bool AutoMonadAction(const ResourcePtr &res);
bool AbstractSpecializeAction(const ResourcePtr &res);
bool GeOptimizeAction(const ResourcePtr &res);
bool VmOptimizeAction(const ResourcePtr &res);
bool PynativeElimOpt(const ResourcePtr &res);
bool TaskEmitAction(const ResourcePtr &res);
bool ExecuteAction(const ResourcePtr &res);
bool StartPSWorkerAction(const ResourcePtr &res);
bool StartFLWorkerAction(const ResourcePtr &res);
bool StartPSServerAction(const ResourcePtr &res);
bool StartPSSchedulerAction(const ResourcePtr &res);
// This action is only for federated learning only. In later version, parameter server mode and federated learning will
// use the same action.
bool StartServerAction(const ResourcePtr &res);
bool DistributedSplitAction(const ResourcePtr &res);

std::vector<ActionItem> GePipeline();
std::vector<ActionItem> VmPipeline(const ResourcePtr &resource);
std::vector<ActionItem> MindIRPipeline();
std::vector<ActionItem> PServerPipeline(const ResourcePtr &resource);
std::vector<ActionItem> ServerPipeline(const ResourcePtr &resource);
std::vector<ActionItem> PSchedulerPipeline(const ResourcePtr &resource);
abstract::AnalysisResult AbstractAnalyze(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_spec, bool clear = false);
FuncGraphPtr ProgramSpecialize(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                               const abstract::AnalysisContextPtr &context);
FuncGraphPtr Renormalize(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                         const abstract::AbstractBasePtrList &args_spec);
void SetRunMode(const FuncGraphPtr &func_graph, compile::Backend *backend_ptr);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_ACTION_H_
