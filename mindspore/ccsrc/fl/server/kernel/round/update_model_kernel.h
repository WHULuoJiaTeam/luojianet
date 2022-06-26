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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_UPDATE_MODEL_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_UPDATE_MODEL_KERNEL_H_

#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "fl/server/common.h"
#include "fl/server/kernel/round/round_kernel.h"
#include "fl/server/kernel/round/round_kernel_factory.h"
#include "fl/server/executor.h"
#include "fl/server/model_store.h"
#ifdef ENABLE_ARMOUR
#include "fl/armour/cipher/cipher_meta_storage.h"
#endif
#include "fl/compression/decode_executor.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// The initial data size sum of federated learning is 0, which will be accumulated in updateModel round.
constexpr uint64_t kInitialDataSizeSum = 0;
// results of signature verification
enum sigVerifyResult { FAILED, TIMEOUT, PASSED };

class UpdateModelKernel : public RoundKernel {
 public:
  UpdateModelKernel() = default;
  ~UpdateModelKernel() override = default;

  void InitKernel(size_t threshold_count) override;
  bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<ps::core::MessageHandler> &message) override;
  bool Reset() override;

  // In some cases, the last updateModel message means this server iteration is finished.
  void OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message) override;

 private:
  ResultCode ReachThresholdForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                          const schema::RequestUpdateModel *update_model_req);
  ResultCode UpdateModel(const schema::RequestUpdateModel *update_model_req, const std::shared_ptr<FBBuilder> &fbb,
                         const DeviceMeta &device_meta);
  std::map<std::string, UploadData> ParseFeatureMap(const schema::RequestUpdateModel *update_model_req);

  void RunAggregation();
  ResultCode CountForAggregation(const std::string &req_fl_id);
  std::map<std::string, UploadData> ParseSignDSFeatureMap(const schema::RequestUpdateModel *update_model_req,
                                                          size_t data_size,
                                                          std::map<std::string, std::vector<float>> *weight_map);
  std::map<std::string, UploadData> ParseUploadCompressFeatureMap(
    const schema::RequestUpdateModel *update_model_req, size_t data_size,
    std::map<std::string, std::vector<float>> *weight_map);
  bool VerifySignDSFeatureMap(const std::unordered_map<std::string, size_t> &model,
                              const schema::RequestUpdateModel *update_model_req);
  bool VerifyUploadCompressFeatureMap(const schema::RequestUpdateModel *update_model_req);
  ResultCode CountForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                 const schema::RequestUpdateModel *update_model_req);
  sigVerifyResult VerifySignature(const schema::RequestUpdateModel *update_model_req);
  void BuildUpdateModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                           const std::string &reason, const std::string &next_req_time);
  ResultCode VerifyUpdateModel(const schema::RequestUpdateModel *update_model_req,
                               const std::shared_ptr<FBBuilder> &fbb, DeviceMeta *device_meta);
  // The executor is for updating the model for updateModel request.
  Executor *executor_{nullptr};

  // The time window of one iteration.
  size_t iteration_time_window_{0};

  // Decode functions of compression.
  std::map<std::string, UploadData> DecodeFeatureMap(std::map<std::string, std::vector<float>> *weight_map,
                                                     const schema::RequestUpdateModel *update_model_req,
                                                     schema::CompressType upload_compress_type, size_t data_size);
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_UPDATE_MODEL_KERNEL_H_
