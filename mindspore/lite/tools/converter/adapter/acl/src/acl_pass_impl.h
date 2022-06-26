/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_IMPL_H
#define TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_IMPL_H

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "backend/common/optimizer/pass.h"
#include "include/errorcode.h"
#include "include/api/types.h"
#include "include/registry/converter_context.h"
#include "cxx_api/model/acl/acl_model_options.h"
#include "tools/converter/adapter/acl/common/acl_types.h"
#include "tools/converter/converter_flags.h"
#include "ops/custom.h"

namespace mindspore {
namespace opt {
using mindspore::converter::FmkType;
using mindspore::lite::STATUS;

class AclPassImpl {
 public:
  explicit AclPassImpl(const converter::Flags &config);
  ~AclPassImpl() = default;

  bool Run(const FuncGraphPtr &func_graph);

 private: /* pre or post pass */
  bool IsDynamicInput();
  STATUS CommonPass(const FuncGraphPtr &func_graph);
  STATUS PreProcGraph(const FuncGraphPtr &func_graph);
  STATUS PostProcGraph(const FuncGraphPtr &func_graph);

 private: /* map func graph */
  STATUS DeparseGraph(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  STATUS RunPrimitiveMapper(const FuncGraphPtr &func_graph);
  std::string AdjustCnodeName(const PrimitivePtr &prim);

 private: /* build func graph */
  STATUS BuildGraph(const FuncGraphPtr &func_graph);
  STATUS ConvertGraphToOm(const FuncGraphPtr &func_graph, Buffer *om_data);
  ParameterPtr CreateOmParameter(const FuncGraphPtr &func_graph, const Buffer &om);
  STATUS CreateGraphAippInput(const FuncGraphPtr &func_graph, const Buffer &om_data);
  STATUS SetAclModelOptions(const FuncGraphPtr &func_graph);
  std::shared_ptr<mindspore::Context> CreateModelContext();
  void SetAclModelInitOptions(const std::shared_ptr<AscendDeviceInfo> &ascend_info);
  void SetAclModelBuildOptions(const std::shared_ptr<AscendDeviceInfo> &ascend_info);

 private: /* create custom node */
  CNodePtr CreateCustomNode(const FuncGraphPtr &func_graph);
  void SetCustomAttrs(const std::shared_ptr<ops::Custom> &prim);
  STATUS SetCustomOutputs(const FuncGraphPtr &func_graph, const CNodePtr &custom_node);
  STATUS SetMultiOutputs(const CNodePtr &new_cnode, std::vector<TypeId> data_type);
  STATUS GetFuncGraphOutputInfo(const FuncGraphPtr &func_graph);
  STATUS TraceOutput(const AnfNodePtr &node);

 private: /* modify graph by custom node */
  STATUS ModifyGraphByCustomNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                 const CNodePtr &custom_node);
  STATUS ReplaceInputsByAippInputs(const FuncGraphPtr &func_graph);

 private:
  FmkType fmk_type_;
  lite::acl::AclModelOptionCfg user_options_cfg_;
  ParameterPtr om_parameter_;
  CNodePtr custom_node_;
  std::shared_ptr<AclModelOptions> options_;
  AnfNodePtrList graph_outputs_;
  AnfNodePtrList graph_aipp_inputs_;
  std::vector<std::string> graph_output_names_;
  std::vector<std::vector<int64_t>> graph_output_dims_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_IMPL_H
