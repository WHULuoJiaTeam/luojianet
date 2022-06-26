/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_COMM_LIB_H_

#include <memory>
#include <vector>
#include <string>
#include "runtime/collective/collective_communication_lib.h"
#include "plugin/device/cpu/hal/hardware/ms_communication_group.h"
#include "distributed/cluster/cluster_context.h"
#include "fl/server/collective_ops_impl.h"

namespace mindspore {
namespace device {
namespace cpu {
constexpr char kMSGlobalGroupName[] = "ms_world_group";
using ClusterContext = mindspore::distributed::cluster::ClusterContext;
using CollectiveOpsImpl = mindspore::fl::server::CollectiveOpsImpl;
using CommunicationGroupInfo = mindspore::fl::server::CommunicationGroupInfo;
using ps::core::NodeCommand;

// The time interval for send info or query info between worker and scheduler.
constexpr uint32_t kWaitDuration = 5;

// The collective communication library for MindSpore self developed communication framework.
class MsCollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static MsCollectiveCommLib &GetInstance() {
    static MsCollectiveCommLib instance;
    return instance;
  }

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size) override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;

  bool AllGatherHostHashName(size_t host_hash_name, std::vector<size_t> *host_hash_names) const override;

  bool BroadcastUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) override;

  bool AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                 const std::string &group_name, void *stream = nullptr) override;

  bool AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                 CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) override {
    return true;
  }

  bool Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type, uint32_t root_rank,
                 const std::string &group_name, void *stream = nullptr) override;

  bool ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, TypeId data_type,
                     CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) override {
    return true;
  }

 private:
  MsCollectiveCommLib();
  ~MsCollectiveCommLib() override = default;

  // Send host hash name to scheduler.
  bool SendHostHashName(size_t host_hash_name) const;

  // Query host hash names of all nodes from scheduler.
  bool QueryHostHashNames(std::vector<size_t> *host_hash_names) const;

  // Send unique id to scheduler.
  bool SendUniqueID(const std::string &group_name, size_t root_info_size, const void *root_info) const;

  // Query unique id from scheduler.
  bool QueryUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) const;

  std::shared_ptr<ps::core::AbstractNode> node_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_COMM_LIB_H_
