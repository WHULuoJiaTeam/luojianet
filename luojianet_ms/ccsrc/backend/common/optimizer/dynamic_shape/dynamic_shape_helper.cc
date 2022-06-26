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

#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"

#include <memory>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "kernel/kernel.h"

namespace luojianet_ms {
namespace opt::dynamic_shape {
bool IsRealCNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n)) {
    CNodePtr cnode = utils::cast<CNodePtr>(n);
    return AnfUtils::IsRealKernel(cnode);
  }
  return false;
}

AnfNodePtr GenInferNode(const AnfNodePtr &node, bool fake_flag) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfUtils::CustomActorCallback actor_func;
  if (fake_flag) {
    actor_func = [](void *) -> void { return; };
  } else {
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    actor_func = [kernel_mod](void *) { kernel_mod->InferOp(); };
  }

  auto infer_node = AnfUtils::NewInferActorNode(actor_func, cnode, fake_flag);
  infer_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return infer_node;
}

AnfNodePtr GenInitNode(const AnfNodePtr &node, bool fake_flag) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfUtils::CustomActorCallback actor_func;
  if (fake_flag) {
    actor_func = [](void *) -> void { return; };
  } else {
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    actor_func = [kernel_mod](void *) { kernel_mod->InitOp(); };
  }

  auto init_node = AnfUtils::NewInitActorNode(actor_func, cnode, fake_flag);
  init_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return init_node;
}

AnfNodePtr GenUpdateNode(const AnfNodePtr &node) {
  // Some not dynamic shape node should sync after launch for latter node.
  // Use a flag `just_sync_flag` to distinguish them with dynamic ones.
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto update_node = AnfUtils::NewUpdateActorNode([kernel_mod](void *) { kernel_mod->UpdateOp(); }, cnode);
  update_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return update_node;
}

bool IsNeedUpdateOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->IsNeedUpdateOp();
}

CustomActorNodeManager &CustomActorNodeManager::Instance() {
  static CustomActorNodeManager instance{};
  return instance;
}
}  // namespace opt::dynamic_shape
}  // namespace luojianet_ms
