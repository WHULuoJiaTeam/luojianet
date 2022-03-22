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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_UB_PATTERN_FUSION_H_
#define LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_UB_PATTERN_FUSION_H_
#include <vector>
#include <string>
#include <map>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "backend/optimizer/ascend/buffer_fusion/fusion_base_pass.h"
#include "ir/anf.h"
#include "backend/optimizer/common/pass.h"
#include "backend/optimizer/common/fusion_id_allocator.h"
#include "runtime/device/kernel_info.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/session/kernel_graph.h"

namespace luojianet_ms {
namespace opt {
using FusedNodeRecord = std::vector<luojianet_ms::HashSet<AnfNodePtr>>;

class UbPatternFusion : public PassWithSwitch {
 public:
  UbPatternFusion() : PassWithSwitch("TbeBufferFusion") {}
  ~UbPatternFusion() override = default;

 private:
  bool RunPass(const FuncGraphPtr &graph) override;

  void GetBufferFusionInfo(session::KernelGraph *kernel_graph,
                           luojianet_ms::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) const;
  bool ReplaceFusionOp(luojianet_ms::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos, int64_t fusion_id,
                       session::KernelGraph *kernel_graph) const;
  bool FuseBufferFusionPattern(session::KernelGraph *kernel_graph) const;
};
}  // namespace opt
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_UB_PATTERN_FUSION_H_
