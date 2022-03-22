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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_BASE_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_BASE_H_

#include <map>
#include <string>
#include <vector>
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_constant.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "register/graph_optimizer/fusion_common/op_slice_info.h"

namespace fe {
enum BufferFusionPassType {
  BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
  BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS,
  CUSTOM_AI_CORE_BUFFER_FUSION_PASS,
  CUSTOM_VECTOR_CORE_BUFFER_FUSION_PASS,
  BUFFER_FUSION_PASS_TYPE_RESERVED
};

class BufferFusionPassBase {
 public:
  explicit BufferFusionPassBase();
  virtual ~BufferFusionPassBase();
  virtual std::vector<BufferFusionPattern *> DefinePatterns() = 0;
  virtual Status GetFusionNodes(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
  virtual Status CalcFusionOpSliceInfo(std::vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info);
  static std::vector<ge::NodePtr> GetMatchedNodes(const BufferFusionMapping &mapping);
  static std::vector<ge::NodePtr> GetMatchedNodesByDescName(const std::string &desc_name,
                                                            const BufferFusionMapping &mapping);
  static ge::NodePtr GetMatchedHeadNode(const std::vector<ge::NodePtr> &matched_nodes);

  void SetName(const std::string &name) { name_ = name; }

  std::string GetName() { return name_; }

 private:
  std::string name_;
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_BASE_H_
