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

#ifndef LUOJIANET_MS_CCSRC_PS_CORE_COMMUNICATOR_SSL_CLIENT_H_
#define LUOJIANET_MS_CCSRC_PS_CORE_COMMUNICATOR_SSL_CLIENT_H_

#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <assert.h>
#include <openssl/pkcs12.h>
#include <openssl/bio.h>

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>

#include "utils/log_adapter.h"
#include "ps/core/comm_util.h"
#include "ps/constants.h"
#include "ps/core/file_configuration.h"
#include "ps/ps_context.h"

namespace luojianet_ms {
namespace ps {
namespace core {
class SSLClient {
 public:
  static SSLClient &GetInstance() {
    static SSLClient instance;
    return instance;
  }
  SSL_CTX *GetSSLCtx() const;

 private:
  SSLClient();
  virtual ~SSLClient();
  SSLClient(const SSLClient &) = delete;
  SSLClient &operator=(const SSLClient &) = delete;

  void InitSSL();
  void CleanSSL();

  void StartCheckCertTime(const Configuration &config, const X509 *cert);
  void StopCheckCertTime();
  void InitSSLCtx(const X509 *cert, const EVP_PKEY *pkey);

  SSL_CTX *ssl_ctx_;
  std::unique_ptr<std::thread> check_time_thread_;
  std::atomic<bool> running_;
  std::atomic<bool> is_ready_;
  std::mutex mutex_;
  std::condition_variable cond_;
};
}  // namespace core
}  // namespace ps
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_PS_CORE_COMMUNICATOR_SSL_CLIENT_H_
