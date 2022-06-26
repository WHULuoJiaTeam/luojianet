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

#ifndef MINDSPORE_CCSRC_PS_UTIL_H_
#define MINDSPORE_CCSRC_PS_UTIL_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include "utils/hash_map.h"
#include "ps/constants.h"
#include "frontend/optimizer/optimizer.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/sparse_optimizer_cpu_kernel.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace ps {
enum ParamType { kUnKnown = 0, kWeight = 1, kAccumulation = 2 };

struct ParamInitInfo {
  std::string param_name_;
  ParamType param_type_{kUnKnown};
  size_t global_seed_{0};
  size_t op_seed_{0};
  float init_val_{0};
};

constexpr size_t kNodeInputWeightNameOffset = 1;
constexpr size_t kNodeInputWeightIndexOffset = 2;

class BACKEND_EXPORT Util {
 public:
  static bool IsRoleOfPServer();
  static bool IsRoleOfScheduler();
  static int64_t optimizer_id(const std::string &name);
  static std::string optimizer_name(int64_t id);
  static std::string optimizer_node_name(int64_t id);
  static bool is_optimizer(const std::string &name);
  static int64_t LocalShard(int64_t first_dim, int64_t rank_id, int64_t server_num);
  static std::map<int64_t, int64_t> AllRankLocalShard(int64_t first_dim, int64_t rank_id, int64_t server_num);
  static void ReduceSparseGradient(float *gradients, int *indices, const size_t indices_size, size_t segment_size,
                                   const size_t first_dim_size, const size_t outer_dim_size,
                                   mindspore::kernel::SparseGradient<int> *unique_sparse_grad);
  static bool FuseServerCommOps(const pipeline::ResourcePtr &res);
  static WeightPtr MakeWeightPtr(const std::shared_ptr<std::vector<float>> &data, bool enable_recovery,
                                 const std::shared_ptr<std::vector<int>> &shape = nullptr);
  static std::string GetPrimitiveName(const CNodePtr &cnode);

 private:
  static void DoFusion(const FuncGraphPtr &func_graph, const std::string &cnode_name,
                       const std::string &fused_cnode_name);
  static kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const std::vector<AnfNodePtr> &node_list);

  static mindspore::HashMap<std::string, int64_t> optimizer_to_ids;
  static mindspore::HashMap<int64_t, std::string> id_to_optimizers;
  static mindspore::HashMap<int64_t, std::string> id_to_optimizer_nodes;
  static int64_t rank_id_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_UTIL_H_
