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

#ifndef GE_GE_LOCAL_ENGINE_ENGINE_GE_LOCAL_ENGINE_H_
#define GE_GE_LOCAL_ENGINE_ENGINE_GE_LOCAL_ENGINE_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <memory>
#include <string>
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"

using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;

namespace ge {
namespace ge_local {
/**
 * ge local engine.
 * Used for the ops not belong to any engine. eg:netoutput
 */
class GE_FUNC_VISIBILITY GeLocalEngine {
 public:
  /**
   * get GeLocalEngine instance.
   * @return  GeLocalEngine instance.
   */
  static GeLocalEngine &Instance();

  virtual ~GeLocalEngine() = default;

  /**
   * When Ge start, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  Status Initialize(const std::map<string, string> &options);

  /**
   * After the initialize, GE will invoke this interface
   * to get the Ops kernel Store.
   * @param ops_kernel_map The ge local's ops kernel info
   */
  void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map);

  /**
   * After the initialize, GE will invoke this interface
   * to get the Graph Optimizer.
   * @param graph_optimizers The ge local's Graph Optimizer objs
   */
  void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers);

  /**
   * When the graph finished, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  Status Finalize();

  // Copy prohibited
  GeLocalEngine(const GeLocalEngine &geLocalEngine) = delete;

  // Move prohibited
  GeLocalEngine(const GeLocalEngine &&geLocalEngine) = delete;

  // Copy prohibited
  GeLocalEngine &operator=(const GeLocalEngine &geLocalEngine) = delete;

  // Move prohibited
  GeLocalEngine &operator=(GeLocalEngine &&geLocalEngine) = delete;

 private:
  GeLocalEngine() = default;

  OpsKernelInfoStorePtr ops_kernel_store_ = nullptr;
};
}  // namespace ge_local
}  // namespace ge

extern "C" {

/**
 * When Ge start, GE will invoke this interface
 * @return The status whether initialize successfully
 */
GE_FUNC_VISIBILITY ge::Status Initialize(const map<string, string> &options);

/**
 * After the initialize, GE will invoke this interface to get the Ops kernel Store
 * @param ops_kernel_map The ge local's ops kernel info
 */
GE_FUNC_VISIBILITY void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map);

/**
 * After the initialize, GE will invoke this interface to get the Graph Optimizer
 * @param graph_optimizers The ge local's Graph Optimizer objs
 */
GE_FUNC_VISIBILITY void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers);

/**
 * When the graph finished, GE will invoke this interface
 * @return The status whether initialize successfully
 */
GE_FUNC_VISIBILITY ge::Status Finalize();
}

#endif  // GE_GE_LOCAL_ENGINE_ENGINE_GE_LOCAL_ENGINE_H_
