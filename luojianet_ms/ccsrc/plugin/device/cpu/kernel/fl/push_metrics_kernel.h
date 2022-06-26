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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_FL_PUSH_METRICS_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_FL_PUSH_METRICS_H_

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "fl/worker/fl_worker.h"

namespace luojianet_ms {
namespace kernel {
// The duration between two PushMetrics requests.
constexpr int kRetryDurationOfPushMetrics = 500;
// Retry for 30 minutes.
constexpr int kMaxRetryTime = 3600;
class PushMetricsKernelMod : public NativeCpuKernelMod {
 public:
  PushMetricsKernelMod() : fbb_(nullptr), total_iteration_(0) {}
  ~PushMetricsKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  void Init(const CNodePtr &kernel_node) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    fbb_ = std::make_shared<fl::FBBuilder>();
    MS_EXCEPTION_IF_NULL(fbb_);
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    output_size_list_.push_back(sizeof(float));

    auto kernel_attr = GetKernelAttrFromNode(kernel_node);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(EXCEPTION) << "PushMetrics does not support this kernel data type: " << kernel_attr;
    }
    kernel_func_ = func_list_[index].second;
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() { return; }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool BuildPushMetricsReq(const std::shared_ptr<fl::FBBuilder> &fbb, T loss, T accuracy) {
    MS_EXCEPTION_IF_NULL(fbb);
    schema::RequestPushMetricsBuilder req_push_metrics_builder(*(fbb.get()));
    req_push_metrics_builder.add_loss(loss);
    req_push_metrics_builder.add_accuracy(accuracy);
    auto req_push_metrics = req_push_metrics_builder.Finish();
    fbb->Finish(req_push_metrics);
    return true;
  }

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                    const std::vector<kernel::AddressPtr> &) {
    if (inputs.size() != 2) {
      MS_LOG(EXCEPTION) << "Input number of PushMetricsKernelMod should be " << 2 << ", but got " << inputs.size();
      return false;
    }

    MS_EXCEPTION_IF_NULL(inputs[0]->addr);
    MS_EXCEPTION_IF_NULL(inputs[1]->addr);
    T loss = *(static_cast<float *>(inputs[0]->addr));
    T accuracy = *(static_cast<float *>(inputs[1]->addr));

    if (!BuildPushMetricsReq(fbb_, loss, accuracy)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
      return false;
    }

    uint32_t retry_time = 0;
    std::shared_ptr<std::vector<unsigned char>> push_metrics_rsp_msg = nullptr;
    do {
      if (!fl::worker::FLWorker::GetInstance().running()) {
        MS_LOG(WARNING) << "Worker has finished.";
        return true;
      }
      retry_time++;
      if (!fl::worker::FLWorker::GetInstance().SendToServer(fl::kLeaderServerRank, fbb_->GetBufferPointer(),
                                                            fbb_->GetSize(), ps::core::TcpUserCommand::kPushMetrics,
                                                            &push_metrics_rsp_msg)) {
        MS_LOG(WARNING) << "Sending request for PushMetrics to server 0 failed.";
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPushMetrics));
        continue;
      } else {
        break;
      }
    } while (retry_time < kMaxRetryTime);

    flatbuffers::Verifier verifier(push_metrics_rsp_msg->data(), push_metrics_rsp_msg->size());
    if (!verifier.VerifyBuffer<schema::ResponsePushMetrics>()) {
      MS_LOG(EXCEPTION) << "The schema of ResponsePushMetrics is invalid.";
      return false;
    }

    const schema::ResponsePushMetrics *push_metrics_rsp =
      flatbuffers::GetRoot<schema::ResponsePushMetrics>(push_metrics_rsp_msg->data());
    MS_EXCEPTION_IF_NULL(push_metrics_rsp);
    auto response_code = push_metrics_rsp->retcode();
    switch (response_code) {
      case schema::ResponseCode_SUCCEED:
      case schema::ResponseCode_OutOfTime:
        break;
      default:
        MS_LOG(WARNING) << "Launching push metrics for worker failed.";
    }

    MS_LOG(INFO) << "Push metrics for loss and accuracy success.";
    fl::worker::FLWorker::GetInstance().SetIterationCompleted();
    return true;
  }

  using PushMetricsFunc =
    std::function<bool(PushMetricsKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, PushMetricsFunc>> func_list_;
  PushMetricsFunc kernel_func_;

  std::shared_ptr<fl::FBBuilder> fbb_;
  size_t total_iteration_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_FL_PUSH_METRICS_H_
