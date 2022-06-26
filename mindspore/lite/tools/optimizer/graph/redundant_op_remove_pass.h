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

#ifndef MINDSPORE_LITE_SRC_PASS_REDUNDANT_OP_REMOVE_PASS_H_
#define MINDSPORE_LITE_SRC_PASS_REDUNDANT_OP_REMOVE_PASS_H_
#include <string>
#include <set>
#include "backend/common/optimizer/pass.h"
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/anf_exporter/fetch_content.h"

using mindspore::converter::FmkType;
namespace mindspore::opt {
class RemoveRedundantOpPass : public Pass {
 public:
  explicit RemoveRedundantOpPass(bool is_train_model)
      : Pass("remove_redundant_op_pass"), is_train_model_(is_train_model) {}
  ~RemoveRedundantOpPass() override = default;
  int ReplaceOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager);
  int ReplaceUpdateStateOp(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node);
  int ReplaceTupleGetItem(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager);
  int RemoveDropoutOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager);
  int RemoveInvalidPadOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager);
  int RemoveInvalidTransposeOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager);
  bool Run(const FuncGraphPtr &graph) override;

 private:
  int GetConstDataFromInputNode(const CNodePtr &cnode, lite::DataInfo *data_info);
  bool is_train_model_ = false;
  std::set<AnfNodePtr> remove_cnode_;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_SRC_PASS_REDUNDANT_OP_REMOVE_PASS_H_
