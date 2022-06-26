/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "fl/server/round.h"
#include <memory>
#include <string>
#include "fl/server/server.h"
#include "fl/server/iteration.h"

namespace mindspore {
namespace fl {
namespace server {
class Server;
class Iteration;
std::atomic<uint32_t> kJobNotReadyPrintTimes = 0;
std::atomic<uint32_t> kJobNotAvailablePrintTimes = 0;
std::atomic<uint32_t> kClusterSafeModePrintTimes = 0;

const uint32_t kPrintTimesThreshold = 3000;
Round::Round(const std::string &name, bool check_timeout, size_t time_window, bool check_count, size_t threshold_count,
             bool server_num_as_threshold)
    : name_(name),
      check_timeout_(check_timeout),
      time_window_(time_window),
      check_count_(check_count),
      threshold_count_(threshold_count),
      server_num_as_threshold_(server_num_as_threshold) {}

void Round::RegisterMsgCallBack(const std::shared_ptr<ps::core::CommunicatorBase> &communicator) {
  MS_EXCEPTION_IF_NULL(communicator);
  MS_LOG(INFO) << "Round " << name_ << " register message callback.";
  communicator->RegisterMsgCallBack(
    name_, [this](std::shared_ptr<ps::core::MessageHandler> message) { LaunchRoundKernel(message); });
}

void Round::Initialize(const TimeOutCb &timeout_cb, const FinishIterCb &finish_iteration_cb) {
  MS_LOG(INFO) << "Round " << name_ << " start initialize.";
  if (check_timeout_) {
    iter_timer_ = std::make_shared<IterationTimer>();
    MS_EXCEPTION_IF_NULL(iter_timer_);

    // 1.Set the timeout callback for the timer.
    iter_timer_->SetTimeOutCallBack([this, timeout_cb](bool, const std::string &) -> void {
      std::string reason = "Round " + name_ + " timeout! This iteration is invalid. Proceed to next iteration.";
      timeout_cb(false, reason);
    });

    // 2.Stopping timer callback which will be set to the round kernel.
    stop_timer_cb_ = [this](void) -> void {
      MS_ERROR_IF_NULL_WO_RET_VAL(iter_timer_);
      MS_LOG(INFO) << "Round " << name_ << " kernel stops its timer.";
      iter_timer_->Stop();
    };
  }

  // Set counter event callbacks for this round if the round kernel is stateful.
  if (check_count_) {
    auto first_count_handler = std::bind(&Round::OnFirstCountEvent, this, std::placeholders::_1);
    auto last_count_handler = std::bind(&Round::OnLastCountEvent, this, std::placeholders::_1);
    DistributedCountService::GetInstance().RegisterCounter(name_, threshold_count_,
                                                           {first_count_handler, last_count_handler});
  }
}

bool Round::ReInitForScaling(uint32_t server_num) {
  // If this round requires up-to-date server number as threshold count, update threshold_count_.
  if (server_num_as_threshold_) {
    MS_LOG(INFO) << "Round " << name_ << " uses up-to-date server number " << server_num << " as its threshold count.";
    threshold_count_ = server_num;
  }
  if (check_count_) {
    auto first_count_handler = std::bind(&Round::OnFirstCountEvent, this, std::placeholders::_1);
    auto last_count_handler = std::bind(&Round::OnLastCountEvent, this, std::placeholders::_1);
    DistributedCountService::GetInstance().RegisterCounter(name_, threshold_count_,
                                                           {first_count_handler, last_count_handler});
  }

  MS_ERROR_IF_NULL_W_RET_VAL(kernel_, false);
  if (name_ == "reconstructSecrets") {
    kernel_->InitKernel(server_num);
  } else {
    kernel_->InitKernel(threshold_count_);
  }
  return true;
}

bool Round::ReInitForUpdatingHyperParams(size_t updated_threshold_count, size_t updated_time_window,
                                         uint32_t server_num) {
  time_window_ = updated_time_window;
  threshold_count_ = updated_threshold_count;
  if (check_count_) {
    if (!DistributedCountService::GetInstance().ReInitCounter(name_, threshold_count_)) {
      MS_LOG(WARNING) << "Reinitializing count for " << name_ << " failed.";
      return false;
    }
  }

  MS_ERROR_IF_NULL_W_RET_VAL(kernel_, false);
  if (name_ == "reconstructSecrets") {
    kernel_->InitKernel(server_num);
  } else {
    kernel_->InitKernel(threshold_count_);
  }
  return true;
}

void Round::BindRoundKernel(const std::shared_ptr<kernel::RoundKernel> &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  kernel_ = kernel;
  kernel_->set_stop_timer_cb(stop_timer_cb_);
  return;
}

void Round::LaunchRoundKernel(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  std::string reason = "";
  if (!IsServerAvailable(&reason)) {
    if (!message->SendResponse(reason.c_str(), reason.size())) {
      MS_LOG(WARNING) << "Sending response failed.";
      return;
    }
    return;
  }

  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  (void)(Iteration::GetInstance().running_round_num_++);
  bool ret = kernel_->Launch(reinterpret_cast<const uint8_t *>(message->data()), message->len(), message);
  // Must send response back no matter what value Launch method returns.
  if (!ret) {
    MS_LOG(DEBUG) << "Launching round kernel of round " + name_ + " failed.";
  }
  (void)(Iteration::GetInstance().running_round_num_--);
  return;
}

void Round::Reset() {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  (void)kernel_->Reset();
}

const std::string &Round::name() const { return name_; }

size_t Round::threshold_count() const { return threshold_count_; }

bool Round::check_timeout() const { return check_timeout_; }

size_t Round::time_window() const { return time_window_; }

void Round::OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  MS_LOG(INFO) << "Round " << name_ << " first count event is triggered.";
  // The timer starts only after the first count event is triggered by DistributedCountService.
  if (check_timeout_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(iter_timer_);
    iter_timer_->Start(std::chrono::milliseconds(time_window_));
  }

  // Some kernels override the OnFirstCountEvent method.
  kernel_->OnFirstCountEvent(message);
  return;
}

void Round::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  MS_LOG(INFO) << "Round " << name_ << " last count event is triggered.";
  // Same as the first count event, the timer must be stopped by DistributedCountService.
  if (check_timeout_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(iter_timer_);
    iter_timer_->Stop();
  }

  // Some kernels override the OnLastCountEvent method.
  kernel_->OnLastCountEvent(message);
  return;
}

bool Round::IsServerAvailable(std::string *reason) {
  MS_ERROR_IF_NULL_W_RET_VAL(reason, false);
  // After one instance is completed, the model should be accessed by clients.
  if (Iteration::GetInstance().instance_state() == InstanceState::kFinish && name_ == "getModel") {
    return true;
  }

  if (!Server::GetInstance().IsReady()) {
    if (kJobNotReadyPrintTimes % kPrintTimesThreshold == 0) {
      MS_LOG(WARNING) << "The server's training job is not ready, please retry " + name_ + " later.";
      kJobNotReadyPrintTimes = 0;
    }
    kJobNotReadyPrintTimes += 1;
    *reason = ps::kJobNotReady;
    return false;
  }

  // If the server state is Disable or Finish, refuse the request.
  if (Iteration::GetInstance().instance_state() == InstanceState::kDisable ||
      Iteration::GetInstance().instance_state() == InstanceState::kFinish) {
    if (kJobNotAvailablePrintTimes % kPrintTimesThreshold == 0) {
      MS_LOG(WARNING) << "The server's training job is disabled or finished, please retry " + name_ + " later.";
      kJobNotAvailablePrintTimes = 0;
    }
    kJobNotAvailablePrintTimes += 1;
    *reason = ps::kJobNotAvailable;
    return false;
  }

  // If the server is still in safemode, reject the request.
  if (Server::GetInstance().IsSafeMode()) {
    if (kClusterSafeModePrintTimes % kPrintTimesThreshold == 0) {
      MS_LOG(WARNING) << "The cluster is still in safemode, please retry " << name_ << " later.";
      kClusterSafeModePrintTimes = 0;
    }
    kClusterSafeModePrintTimes += 1;
    *reason = ps::kClusterSafeMode;
    return false;
  }
  return true;
}

void Round::KernelSummarize() {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  (void)kernel_->Summarize();
}

size_t Round::kernel_total_client_num() const { return kernel_->total_client_num(); }

size_t Round::kernel_accept_client_num() const { return kernel_->accept_client_num(); }

size_t Round::kernel_reject_client_num() const { return kernel_->reject_client_num(); }

void Round::InitkernelClientVisitedNum() { kernel_->InitClientVisitedNum(); }

void Round::InitkernelClientUploadLoss() { kernel_->InitClientUploadLoss(); }

float Round::kernel_upload_loss() const { return kernel_->upload_loss(); }
}  // namespace server
}  // namespace fl
}  // namespace mindspore
