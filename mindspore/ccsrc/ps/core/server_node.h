/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_CORE_SERVER_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_SERVER_NODE_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

#include "ps/core/cluster_metadata.h"
#include "ps/core/cluster_config.h"
#include "ps/ps_context.h"
#include "ps/core/communicator/tcp_client.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/abstract_node.h"
#include "ps/core/communicator/task_executor.h"
#include "ps/core/communicator/communicator_base.h"

namespace mindspore {
namespace ps {
namespace core {
class ServerNode : public AbstractNode {
 public:
  ServerNode() = default;

  ~ServerNode() override = default;

  bool Start(const uint32_t &timeout = PSContext::instance()->cluster_config().cluster_available_timeout) override;
  bool Stop() override;
  bool Finish(const uint32_t &timeout = kTimeoutInSeconds) override;

 private:
  void Initialize();
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_SERVER_NODE_H_
