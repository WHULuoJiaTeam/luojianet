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

#ifndef LUOJIANET_MS_CCSRC_PS_CORE_COMMUNICATOR_HTTP_MSG_HANDLER_H_
#define LUOJIANET_MS_CCSRC_PS_CORE_COMMUNICATOR_HTTP_MSG_HANDLER_H_

#include <memory>
#include "ps/core/communicator/http_message_handler.h"
#include "ps/core/communicator/message_handler.h"

namespace luojianet_ms {
namespace ps {
namespace core {
constexpr int kHttpSuccess = 200;
class HttpMsgHandler : public MessageHandler {
 public:
  explicit HttpMsgHandler(const std::shared_ptr<HttpMessageHandler> &http_msg);
  ~HttpMsgHandler() override = default;

  void *data() const override;
  size_t len() const override;
  bool SendResponse(const void *data, const size_t &len) override;

 private:
  std::shared_ptr<HttpMessageHandler> http_msg_;
  unsigned char *data_;
  size_t len_;
};
}  // namespace core
}  // namespace ps
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_PS_CORE_COMMUNICATOR_HTTP_MSG_HANDLER_H_
