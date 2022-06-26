/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_VM_SEGMENT_RUNNER_H_
#define MINDSPORE_CCSRC_VM_SEGMENT_RUNNER_H_

#include <vector>
#include <string>
#include <tuple>
#include <set>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "backend/graph_compiler/vmimpl.h"
#include "backend/graph_compiler/graph_partition.h"

namespace mindspore {
namespace compile {
struct LinConvertResult {
  RunFuncPtr run;
  RunFuncPtr simu_run;
  std::vector<AnfNodePtr> inputs;
  std::vector<AnfNodePtr> outputs;
  uint32_t graph_id;
};

using LinkFuncType = std::function<LinConvertResult(const GraphSegmentPtr &, const std::string &)>;
extern LinkFuncType MsVmConvert;
extern std::set<std::string> backend_list;
std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> TransformSegmentToAnfGraph(const AnfNodePtrList &lst);
}  // namespace compile
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_VM_SEGMENT_RUNNER_H_
