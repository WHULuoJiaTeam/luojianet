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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_FUSION_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_FUSION_PASS_H_
#include <vector>
#include "src/delegate/npu/op/npu_op.h"
#include "src/delegate/npu/pass/npu_base_pass.h"

namespace mindspore {
class NPUFusionPass : public NPUBasePass {
 public:
  NPUFusionPass() { name_ = "NPUFusionPass"; }

  int Run(NPUGraph *subgraph) override;

 protected:
  int UpdatePreOps(NPUOp *cur_op);
  int UpdatePostOps(NPUOp *cur_op);
  void RemoveAndFreeOp(NPUOp *cur_op);
  int UpdateOp(NPUOp *cur_op);
  int CommonFusion(NPUOp *cur_op);
  int FormatFusion(NPUOp *cur_op);

 private:
  std::vector<NPUOp *> *all_ops_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_FUSION_PASS_H_
