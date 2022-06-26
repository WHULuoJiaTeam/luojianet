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

#ifndef LUOJIANET_MS_CCSRC_RUNTIME_FRAMEWORK_ACTO_RPC_RPC_ACTOR_H_
#define LUOJIANET_MS_CCSRC_RUNTIME_FRAMEWORK_ACTO_RPC_RPC_ACTOR_H_

#include <set>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "distributed/cluster/cluster_context.h"
#include "distributed/rpc/tcp/tcp_client.h"
#include "distributed/rpc/tcp/tcp_server.h"

namespace luojianet_ms {
namespace runtime {
using distributed::cluster::ActorRouteTableProxy;
using distributed::cluster::ActorRouteTableProxyPtr;
using distributed::cluster::ClusterContext;
using distributed::rpc::TCPClient;
using distributed::rpc::TCPServer;
using luojianet_ms::device::KernelInfo;
using ps::core::ActorAddress;

// The inter-process edge mark between two nodes.
constexpr char kInterProcessEdgeMark[] = "->";

// RpcActor is used to do rpc with other processes in distributed execution.
// Besides data arrows and controlling arrows, RpcActor also has inter-process arrows which is in charge of remote
// communication with other processes. It supports both sync and async communication.
class RpcActor : public KernelActor {
 public:
  explicit RpcActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                    const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                    GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                    const std::set<size_t> &modifiable_ref_output_indexes, const KernelTransformType &type)
      : KernelActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                    modifiable_ref_input_indexes, modifiable_ref_output_indexes, type),
        input_inter_process_num_(0) {}
  ~RpcActor() override = default;

  // Normally, an actor's op_context is passed by its input actor, but rpc actors could be triggered by inter-process
  // arrows which do not contain op_context. So we need to set op_context manually.
  virtual void SetOpcontext(OpContext<DeviceTensor> *const op_context);

  // Reset op context. Because op context is recreated for each each sinked loop, this method should be called after
  // each sinked loop is done in case rpc actors visit the invalid op context.
  virtual void ResetOpcontext() {}

  // Set the actor route proxy for rpc actors.
  void SetActorRouteRableProxy(const ActorRouteTableProxyPtr &proxy);

  // Set the inter-process edge name for rpc actor.
  void SetInterProcessEdgeName(const std::string &src_node_name, const std::string &dst_node_name);

  // Set some info which will be used for rpc routing.
  virtual void SetRouteInfo(uint32_t peer_rank, const std::string &peer_role, const std::string &src_node_name,
                            const std::string &dst_node_name) {}

 protected:
  // The op context to run rpc actor inter-process op. Set by method 'SetOpcontext'.
  OpContext<DeviceTensor> *op_context_;

  // The inter-process edge name. It is also used as the actor id for route. It's a string consists of source node name
  // and destination node name. The format is "source node name"->"destination node name". For each inter-process edge,
  // this is is unique. Rpc actor with the same inter_process_edge_name_ should not be in the same process.
  std::string inter_process_edge_name_;

  // The node name of rpc actor's peers. They are not the name of send or recv nodes. Instead, they are the names of the
  // nodes which use send node as output and recv node as input.
  std::vector<std::string> rpc_input_node_name_;
  std::vector<std::string> rpc_output_node_name_;

  // The iter-process inputs number. This should be the same as size of vector rpc_input_node_name_.
  size_t input_inter_process_num_;

  // The inter-process inputs of each sequential number.
  luojianet_ms::HashMap<int, std::vector<std::string>> input_op_inter_process_;

  // The arrows represent inter-process communication.
  std::vector<AID> inter_process_input_arrows_;
  std::vector<AID> inter_process_output_arrows_;

  ActorRouteTableProxyPtr actor_route_table_proxy_;

 private:
  friend class GraphScheduler;
};

using RpcActorPtr = std::shared_ptr<RpcActor>;
}  // namespace runtime
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_RUNTIME_FRAMEWORK_ACTO_RPC_RPC_ACTOR_H_
