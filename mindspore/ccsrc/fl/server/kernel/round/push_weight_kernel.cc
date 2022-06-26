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

#include "fl/server/kernel/round/push_weight_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void PushWeightKernel::InitKernel(size_t) {
  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }
  local_rank_ = DistributedCountService::GetInstance().local_rank();
}

bool PushWeightKernel::Launch(const uint8_t *req_data, size_t len,
                              const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_LOG(INFO) << "Launching PushWeightKernel kernel.";
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, reason.c_str(), reason.size());
    return true;
  }

  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestPushWeight>()) {
    std::string reason = "The schema of RequestPushWeight is invalid.";
    BuildPushWeightRsp(fbb, schema::ResponseCode_RequestError, reason, LocalMetaStore::GetInstance().curr_iter_num());
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  const schema::RequestPushWeight *push_weight_req = flatbuffers::GetRoot<schema::RequestPushWeight>(req_data);
  if (push_weight_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestPushWeight";
    BuildPushWeightRsp(fbb, schema::ResponseCode_RequestError, reason, LocalMetaStore::GetInstance().curr_iter_num());
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  ResultCode result_code = PushWeight(fbb, push_weight_req);
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  if (result_code != ResultCode::kSuccess) {
    return false;
  }
  return true;
}

bool PushWeightKernel::Reset() {
  MS_LOG(INFO) << "PushWeightKernel reset!";
  StopTimer();
  DistributedCountService::GetInstance().ResetCounter(name_);
  return true;
}

void PushWeightKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {
  if (ps::PSContext::instance()->resetter_round() == ps::ResetterRound::kPushWeight) {
    FinishIteration(true);
  }
  return;
}

ResultCode PushWeightKernel::PushWeight(const std::shared_ptr<FBBuilder> &fbb,
                                        const schema::RequestPushWeight *push_weight_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(fbb, ResultCode::kFail);
  MS_ERROR_IF_NULL_W_RET_VAL(push_weight_req, ResultCode::kFail);
  size_t iteration = IntToSize(push_weight_req->iteration());
  size_t current_iter = LocalMetaStore::GetInstance().curr_iter_num();
  if (iteration != current_iter) {
    std::string reason = "PushWeight iteration number is invalid:" + std::to_string(iteration) +
                         ", current iteration:" + std::to_string(current_iter);
    BuildPushWeightRsp(fbb, schema::ResponseCode_SucNotReady, reason, current_iter);
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }

  std::map<std::string, Address> upload_feature_map = ParseFeatureMap(push_weight_req);
  if (upload_feature_map.empty()) {
    std::string reason = "PushWeight feature_map is empty.";
    BuildPushWeightRsp(fbb, schema::ResponseCode_RequestError, reason, current_iter);
    MS_LOG(ERROR) << reason;
    return ResultCode::kFail;
  }

  if (!executor_->HandlePushWeight(upload_feature_map)) {
    std::string reason = "Pushing weight failed.";
    BuildPushWeightRsp(fbb, schema::ResponseCode_SucNotReady, reason, current_iter);
    MS_LOG(ERROR) << reason;
    return ResultCode::kFail;
  }
  MS_LOG(INFO) << "Pushing weight for iteration " << current_iter << " succeeds.";

  if (!DistributedCountService::GetInstance().Count(name_, std::to_string(local_rank_))) {
    std::string reason = "Count for push weight request failed.";
    BuildPushWeightRsp(fbb, schema::ResponseCode_SystemError, reason, current_iter);
    MS_LOG(ERROR) << reason;
    return ResultCode::kFail;
  }
  BuildPushWeightRsp(fbb, schema::ResponseCode_SUCCEED, "PushWeight succeed.", current_iter);
  return ResultCode::kSuccess;
}

std::map<std::string, Address> PushWeightKernel::ParseFeatureMap(const schema::RequestPushWeight *push_weight_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(push_weight_req, {});
  std::map<std::string, Address> upload_feature_map;
  auto fbs_feature_map = push_weight_req->feature_map();
  MS_ERROR_IF_NULL_W_RET_VAL(fbs_feature_map, upload_feature_map);
  for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
    std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
    float *weight_data = const_cast<float *>(fbs_feature_map->Get(i)->data()->data());
    size_t weight_size = fbs_feature_map->Get(i)->data()->size() * sizeof(float);
    upload_feature_map[weight_full_name] = {weight_data, weight_size};
  }
  return upload_feature_map;
}

void PushWeightKernel::BuildPushWeightRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                          const std::string &reason, size_t iteration) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  schema::ResponsePushWeightBuilder rsp_push_weight_builder(*(fbb.get()));
  rsp_push_weight_builder.add_retcode(SizeToInt(retcode));
  rsp_push_weight_builder.add_reason(fbs_reason);
  rsp_push_weight_builder.add_iteration(SizeToInt(iteration));
  auto rsp_push_weight = rsp_push_weight_builder.Finish();
  fbb->Finish(rsp_push_weight);
  return;
}

REG_ROUND_KERNEL(pushWeight, PushWeightKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
