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

#ifndef MINDSPORE_CCSRC_PS_CORE_LEADER_SCALER_H_
#define MINDSPORE_CCSRC_PS_CORE_LEADER_SCALER_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ps/core/communicator/tcp_client.h"
#include "ps/core/node_manager.h"
#include "ps/core/node.h"
#include "ps/core/communicator/request_process_result_code.h"
#include "ps/constants.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {
// The class helps scheduler node to do scale out/in operation for the cluster.
class LeaderScaler {
 public:
  explicit LeaderScaler(Node *const node) : node_(node) {}
  ~LeaderScaler() = default;

  // When the scheduler receives the scale out message, it will send this message to the workers and servers.
  void ScaleOutAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager);
  // When the scheduler receives the scale in message, it will send this message to the workers and servers.
  void ScaleInAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager, bool is_node_scale_in);

  void ScaleOutRollbackAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager);

 private:
  // The node_ will only be instantiated with scheduler node.
  Node *const node_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_LEADER_SCALER_H_
