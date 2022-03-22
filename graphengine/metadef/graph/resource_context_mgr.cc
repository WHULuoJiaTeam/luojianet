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

#include "graph/resource_context_mgr.h"
#include "graph/debug/ge_log.h"

namespace ge {
ResourceContext *ResourceContextMgr::GetResourceContext(const std::string &resource_key) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  const auto iter = resource_keys_to_contexts_.find(resource_key);
  if (iter == resource_keys_to_contexts_.end()) {
    return nullptr;
  }
  return resource_keys_to_contexts_[resource_key].get();
}

graphStatus ResourceContextMgr::SetResourceContext(const std::string &resource_key, ResourceContext *const context) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  resource_keys_to_contexts_[resource_key] = std::unique_ptr<ResourceContext>(context);
  return GRAPH_SUCCESS;
}

graphStatus ResourceContextMgr::RegisterNodeReliedOnResource(const std::string &resource_key, NodePtr &node) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  (void)resource_keys_to_read_nodes_[resource_key].emplace(node);
  return GRAPH_SUCCESS;
}

std::unordered_set<NodePtr> &ResourceContextMgr::MutableNodesReliedOnResource(const std::string &resource_key) {
  const std::lock_guard<std::mutex> lk(ctx_mu_);
  return resource_keys_to_read_nodes_[resource_key];
}

graphStatus ResourceContextMgr::ClearContext() {
  const std::lock_guard<std::mutex> lk_resource(ctx_mu_);
  resource_keys_to_contexts_.clear();
  resource_keys_to_read_nodes_.clear();
  return GRAPH_SUCCESS;
}
}  // namespace ge
