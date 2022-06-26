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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_START_FL_JOB_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_START_FL_JOB_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fl/server/common.h"
#include "fl/server/executor.h"
#include "fl/server/kernel/round/round_kernel.h"
#include "fl/server/kernel/round/round_kernel_factory.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "fl/compression/encode_executor.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
class StartFLJobKernel : public RoundKernel {
 public:
  StartFLJobKernel() : executor_(nullptr), iteration_time_window_(0), iter_next_req_timestamp_(0) {}
  ~StartFLJobKernel() override = default;

  void InitKernel(size_t threshold_count) override;
  bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<ps::core::MessageHandler> &message) override;
  bool Reset() override;

  void OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message) override;

 private:
  // Returns whether the startFLJob count of this iteration has reached the threshold.
  ResultCode ReachThresholdForStartFLJob(const std::shared_ptr<FBBuilder> &fbb,
                                         const schema::RequestFLJob *start_fl_job_req);

  // The metadata of device will be stored and queried in updateModel round.
  DeviceMeta CreateDeviceMetadata(const schema::RequestFLJob *start_fl_job_req);

  // Returns whether the request is valid for startFLJob.For now, the condition is simple. We will add more conditions
  // to device in later versions.
  ResultCode ReadyForStartFLJob(const std::shared_ptr<FBBuilder> &fbb, const DeviceMeta &device_meta);

  // Distributed count service counts for startFLJob.
  ResultCode CountForStartFLJob(const std::shared_ptr<FBBuilder> &fbb, const schema::RequestFLJob *start_fl_job_req);

  void StartFLJob(const std::shared_ptr<FBBuilder> &fbb, const DeviceMeta &device_meta,
                  const schema::RequestFLJob *start_fl_job_req);

  bool JudgeFLJobCert(const std::shared_ptr<FBBuilder> &fbb, const schema::RequestFLJob *start_fl_job_req);

  bool StoreKeyAttestation(const std::shared_ptr<FBBuilder> &fbb, const schema::RequestFLJob *start_fl_job_req);

  // Build response for startFLJob round no matter success or failure.
  void BuildStartFLJobRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                          const std::string &reason, const bool is_selected, const std::string &next_req_time,
                          const std::map<std::string, AddressPtr> &feature_maps = {},
                          const schema::CompressType &compressType = schema::CompressType_NO_COMPRESS,
                          const std::map<std::string, AddressPtr> &compress_feature_maps = {});

  // The executor is for getting the initial model for startFLJob request.
  Executor *executor_;

  // The time window of one iteration.
  size_t iteration_time_window_;

  // Timestamp of next request time for this iteration.
  uint64_t iter_next_req_timestamp_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_START_FL_JOB_KERNEL_H_
