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
#ifndef INC_GRAPH_RESOURCE_CONTEXT_MRG_H_
#define INC_GRAPH_RESOURCE_CONTEXT_MRG_H_

#include <string>
#include <map>
#include <mutex>
#include "external/graph/resource_context.h"
#include "graph/ge_error_codes.h"
#include "graph/node.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ResourceContextMgr {
 public:
  ResourceContextMgr() = default;
  ~ResourceContextMgr() = default;
  /**
   * Given resource_key , return corresponding resource pointer
   * @param resource_key
   * @return orresponding resource pointer
   */
  ResourceContext *GetResourceContext(const std::string &resource_key);
  /**
   * Given resource_key , corresponding resource pointer, set resouce_context with new resource
   * @param resource_key
   * @param context
   * @return status
   */
  graphStatus SetResourceContext(const std::string &resource_key, ResourceContext *const context);
  /**
   * Given resource_key , node reiled on this resource, mgr will keep the relation
   * @param resource_key
   * @param node
   * @return status
   */
  graphStatus RegisterNodeReliedOnResource(const std::string &resource_key, NodePtr &node);
  /**
   * Given resource_key , mgr find node reiled on this reousrce.
   * @param resource_key
   * @param read_nodes
   * @return status
   */
  std::unordered_set<NodePtr> &MutableNodesReliedOnResource(const std::string &resource_key);
  /**
   * Resource context need to be cleared when session finalize
   * @return status
   */
  graphStatus ClearContext();
  
 private:
  std::mutex ctx_mu_;
  std::map<std::string, std::unique_ptr<ResourceContext>> resource_keys_to_contexts_;
  std::map<std::string, std::unordered_set<NodePtr>> resource_keys_to_read_nodes_;
};
}  // namespace ge
#endif  //  INC_GRAPH_RESOURCE_CONTEXT_MRG_H_
