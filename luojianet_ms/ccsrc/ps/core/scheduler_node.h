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

#ifndef LUOJIANET_MS_CCSRC_PS_CORE_SCHEDULER_NODE_H_
#define LUOJIANET_MS_CCSRC_PS_CORE_SCHEDULER_NODE_H_

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <condition_variable>

#include "ps/core/cluster_config.h"
#include "ps/ps_context.h"
#include "ps/core/communicator/tcp_client.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/node_manager.h"
#include "ps/core/node.h"
#include "ps/core/communicator/request_process_result_code.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/constants.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/communicator/http_server.h"
#include "ps/core/leader_scaler.h"
#include "ps/core/recovery_base.h"
#include "ps/core/instance_manager.h"
#include "distributed/cluster/actor_route_table_service.h"
#include "include/backend/visible.h"

namespace luojianet_ms {
namespace ps {
namespace core {
using distributed::cluster::ActorRouteTableService;
class BACKEND_EXPORT SchedulerNode : public Node {
 public:
  SchedulerNode()
      : server_(nullptr),
        scheduler_thread_(nullptr),
        update_state_thread_(nullptr),
        update_persistent_cmd_thread_(nullptr),
        restful_thread_(nullptr),
        http_server_(nullptr),
        client_thread_(nullptr),
        is_client_started_(false),
        leader_scaler_(nullptr),
        scheduler_recovery_(nullptr),
        persistent_cmd_(PersistentCommand::DEFAULT),
        is_worker_timeout_(false),
        actor_route_table_service_(nullptr) {}
  ~SchedulerNode() override;

  typedef void (SchedulerNode::*ResponseHandler)(const std::shared_ptr<TcpServer> &server,
                                                 const std::shared_ptr<TcpConnection> &conn,
                                                 const std::shared_ptr<MessageMeta> &meta, const void *data,
                                                 size_t size);

  bool Start(const uint32_t &timeout = PSContext::instance()->cluster_config().cluster_available_timeout) override;
  bool Stop() override;
  bool Finish(const uint32_t &timeout = kTimeoutInSeconds) override;

 protected:
  void Initialize();

  void InitCommandHandler();
  void CreateTcpServer();
  void StartUpdateClusterStateTimer();
  // Persistent timer, periodically trigger persistent behavior.
  void StartUpdatePersistentCommandTimer();

  // Register and initialize the actor route table service.
  void RegisterActorRouteTableServiceHandler();
  void InitializeActorRouteTableService();

  // Register collective communication initialization service.
  virtual void RegisterInitCollectCommServiceHandler() {}

  // Register recovery service.
  virtual void RegisterRecoveryServiceHandler() {}

  // Handle node timeout info and update nodes which finish transform graph.
  virtual void HandleNodeTimeoutForRecovery(const std::unordered_map<std::string, NodeInfo> &timeout_nodes_infos) {}

  // Recover finish transform nodes info when nodes recover heartbeat.
  virtual void HandleNodeRecoverByHeartBeat(uint32_t rank_id) {}

  // Disaster recovery for operation aspects in PS mode.
  virtual void RecoverFromPersistence() {}

  const std::shared_ptr<TcpClient> &GetOrCreateClient(const NodeInfo &node_info);

  void ProcessHeartbeat(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                        const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessRegister(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                       const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessFinish(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                     const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessFetchMetadata(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                            const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process scale_out_done messages from workers/servers
  void ProcessScaleOutDone(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                           const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  // Process scale_in_done messages from workers/servers
  void ProcessScaleInDone(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                          const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process scale_in_done messages from workers/servers
  void ProcessSendEvent(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                        const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process register actor route messages from other nodes.
  void ProcessRegisterActorRoute(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                                 const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process delete actor route messages from other nodes.
  void ProcessDeleteActorRoute(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                               const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process lookup actor route messages from other nodes.
  void ProcessLookupActorRoute(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                               const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Determine whether the registration request of the node should be rejected, the registration of the
  // alive node should be rejected.
  virtual bool NeedRejectRegister(const NodeInfo &node_info) { return false; }

  // After scheduler collects all registered message, it actively sends finish to the node connected by the client.
  void SendMetadata(const std::shared_ptr<TcpClient> &client, uint32_t rank_id);
  // After scheduler collects all finish message, it actively sends finish to the node connected by the client.
  void SendFinish(const std::shared_ptr<TcpClient> &client);

  // After scheduler collects all scale_out_done message, it actively sends scale_out_done to the node connected by the
  // client.
  void SendScaleOutDone(const std::shared_ptr<TcpClient> &client);

  // After scheduler collects all scale_in_done message, it actively sends scale_out_done to the node connected by the
  // client.
  void SendScaleInDone(const std::shared_ptr<TcpClient> &client);
  // After scheduler receive SEND_EVENT message, it will broadcast the event to all other nodes.
  void SendEvent(const std::shared_ptr<TcpClient> &client, const uint32_t &event);

  // Handle the scale out http request, then delegate to the leader scaler to process scale out asynchronously.
  void ProcessScaleOut(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the scale in http request, then delegate to the leader scaler to process scale in asynchronously.
  void ProcessScaleIn(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the get nodes info http request Synchronously.
  void ProcessGetNodesInfo(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the get cluster state http request Synchronously.
  void ProcessGetClusterState(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the new instance http request Synchronously.
  void ProcessNewInstance(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the query instance http request Synchronously.
  void ProcessQueryInstance(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the enable FLS http request Synchronously.
  void ProcessEnableFLS(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the disable FLS http request Synchronously.
  void ProcessDisableFLS(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the scale out rollback http request, then delegate to the leader scaler to
  // process scale out rollback asynchronously.
  void ProcessScaleOutRollback(const std::shared_ptr<HttpMessageHandler> &resp);

  bool QueryNodeScaleState(const std::shared_ptr<HttpMessageHandler> &resp);

  // check whether the cluster is in the ready state.
  RequestProcessResult CheckIfClusterReady();

  // check whether the node id is legal.
  RequestProcessResult CheckIfNodeIdLegal(const std::vector<std::string> &node_ids);

  void StartRestfulServer(const std::string &address, std::uint16_t port, size_t thread_num = 10);

  void StopRestfulServer();

  void InitNodeMetaData();

  bool RecoverScheduler();

  void PersistMetaData();

  bool CheckIfNodeDisconnected() const;

  virtual void RunRecovery();

  void BroadcastTimeoutEvent();

  void SetRegisterConnectionFd(const std::shared_ptr<TcpConnection> &conn, const std::string &node_id);

  virtual bool SendPrepareBuildingNetwork(const std::unordered_map<std::string, NodeInfo> &node_infos);

  // Responding peer with the general response message.
  void GeneralResponse(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                       const std::shared_ptr<MessageMeta> &meta, bool is_success, const std::string &error);

  bool BuildingNetwork();

  std::shared_ptr<TcpServer> server_;
  std::unique_ptr<std::thread> scheduler_thread_;
  std::unique_ptr<std::thread> update_state_thread_;
  std::unique_ptr<std::thread> update_persistent_cmd_thread_;
  std::condition_variable persistent_cmd_cv_;
  std::mutex persistent_cmd_mutex_;

  luojianet_ms::HashMap<NodeCommand, ResponseHandler> handlers_;

  NodeManager node_manager_;

  // In this thread will start a http server.
  std::unique_ptr<std::thread> restful_thread_;
  std::shared_ptr<HttpServer> http_server_;

  std::unordered_map<std::string, std::shared_ptr<TcpClient>> connected_nodes_;

  std::shared_ptr<TcpClient> client_to_scheduler_;
  std::unique_ptr<std::thread> client_thread_;
  std::atomic<bool> is_client_started_;

  std::unique_ptr<LeaderScaler> leader_scaler_;

  std::unordered_map<std::string, OnRequestReceive> callbacks_;

  // Used to persist and obtain metadata information for scheduler.
  std::unique_ptr<RecoveryBase> scheduler_recovery_;
  // persistent command need to be sent.
  std::atomic<PersistentCommand> persistent_cmd_;

  // The node id of scale in nodes.
  std::vector<std::string> scale_in_node_ids_;

  std::unique_ptr<InstanceManager> instance_manager_;

  std::atomic<bool> is_worker_timeout_;
  // This is a map of register connection fd to client node id
  std::unordered_map<int, std::string> register_connection_fd_;

  std::unique_ptr<ActorRouteTableService> actor_route_table_service_;
};
}  // namespace core
}  // namespace ps
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_PS_CORE_SCHEDULER_NODE_H_
