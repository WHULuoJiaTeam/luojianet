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

#include "ps/core/instance_manager.h"

namespace luojianet_ms {
namespace ps {
namespace core {
void InstanceManager::NewInstanceAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &,
                                       const std::string &body, const uint64_t &request_id, const NodeInfo &node_info) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(node_);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info.rank_id_);
  message_meta->set_role(node_info.node_role_);
  message_meta->set_user_cmd(static_cast<int32_t>(TcpUserCommand::kNewInstance));

  if (!client->SendMessage(message_meta, Protos::RAW, body.data(), body.length())) {
    MS_LOG(WARNING) << "Send new instance timeout!";
  }

  MS_LOG(INFO) << "The scheduler is sending new instance to " << node_info.node_id_;
}

void InstanceManager::QueryInstanceAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &,
                                         const uint64_t &request_id, const NodeInfo &node_info) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(node_);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info.rank_id_);
  message_meta->set_role(node_info.node_role_);
  message_meta->set_user_cmd(static_cast<int32_t>(TcpUserCommand::kQueryInstance));

  std::string res;
  if (!client->SendMessage(message_meta, Protos::RAW, res.data(), res.length())) {
    MS_LOG(WARNING) << "Send query instance timeout!";
  }

  MS_LOG(INFO) << "The scheduler is sending query instance to " << node_info.node_id_;
}

void InstanceManager::EnableFLSAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &,
                                     const uint64_t &request_id, const NodeInfo &node_info) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(node_);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info.rank_id_);
  message_meta->set_role(node_info.node_role_);
  message_meta->set_user_cmd(static_cast<int32_t>(TcpUserCommand::kEnableFLS));

  std::string res;
  if (!client->SendMessage(message_meta, Protos::RAW, res.data(), res.length())) {
    MS_LOG(WARNING) << "Send query instance timeout!";
  }

  MS_LOG(INFO) << "The scheduler is sending enable FLS to " << node_info.node_id_;
}

void InstanceManager::DisableFLSAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &,
                                      const uint64_t &request_id, const NodeInfo &node_info) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(node_);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info.rank_id_);
  message_meta->set_role(node_info.node_role_);
  message_meta->set_user_cmd(static_cast<int32_t>(TcpUserCommand::kDisableFLS));

  std::string res;
  if (!client->SendMessage(message_meta, Protos::RAW, res.data(), res.length())) {
    MS_LOG(WARNING) << "Send query instance timeout!";
  }

  MS_LOG(INFO) << "The scheduler is sending disable FLS to " << node_info.node_id_;
}

void InstanceManager::QueryNodeScaleState(const std::shared_ptr<TcpClient> &client, const NodeManager &,
                                          const uint64_t &request_id, const NodeInfo &node_info) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(node_);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info.rank_id_);
  message_meta->set_role(node_info.node_role_);
  message_meta->set_user_cmd(static_cast<int32_t>(TcpUserCommand::kQueryNodeScaleState));

  std::string res;
  if (!client->SendMessage(message_meta, Protos::RAW, res.data(), res.length())) {
    MS_LOG(WARNING) << "Send query node scale state timeout!";
  }

  MS_LOG(INFO) << "The scheduler is sending query node scale state to " << node_info.node_id_;
}
}  // namespace core
}  // namespace ps
}  // namespace luojianet_ms
