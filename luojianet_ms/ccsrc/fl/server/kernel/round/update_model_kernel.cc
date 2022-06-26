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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "fl/server/kernel/round/update_model_kernel.h"

namespace luojianet_ms {
namespace fl {
namespace server {
namespace kernel {
const char *kCountForAggregation = "count_for_aggregation";

void UpdateModelKernel::InitKernel(size_t threshold_count) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  InitClientVisitedNum();
  InitClientUploadLoss();
  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }

  PBMetadata client_list;
  DistributedMetadataStore::GetInstance().RegisterMetadata(kCtxUpdateModelClientList, client_list);
  LocalMetaStore::GetInstance().put_value(kCtxUpdateModelThld, threshold_count);
  LocalMetaStore::GetInstance().put_value(kCtxFedAvgTotalDataSize, kInitialDataSizeSum);

  auto first_cnt_handler = [](std::shared_ptr<ps::core::MessageHandler>) {};
  auto last_cnt_handler = [this](std::shared_ptr<ps::core::MessageHandler>) { RunAggregation(); };
  DistributedCountService::GetInstance().RegisterCounter(kCountForAggregation, threshold_count,
                                                         {first_cnt_handler, last_cnt_handler});
}

bool UpdateModelKernel::Launch(const uint8_t *req_data, size_t len,
                               const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_LOG(DEBUG) << "Launching UpdateModelKernel kernel.";

  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(WARNING) << reason;
    SendResponseMsg(message, reason.c_str(), reason.size());
    return false;
  }

  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestUpdateModel>()) {
    std::string reason = "The schema of RequestUpdateModel is invalid.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  const schema::RequestUpdateModel *update_model_req = flatbuffers::GetRoot<schema::RequestUpdateModel>(req_data);
  if (update_model_req == nullptr || update_model_req->fl_id() == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestUpdateModel.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  ResultCode result_code = ReachThresholdForUpdateModel(fbb, update_model_req);
  if (result_code != ResultCode::kSuccess) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  DeviceMeta device_meta;
  result_code = VerifyUpdateModel(update_model_req, fbb, &device_meta);
  if (result_code != ResultCode::kSuccess) {
    MS_LOG(DEBUG) << "Verify updating model failed.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  result_code = CountForUpdateModel(fbb, update_model_req);
  if (result_code != ResultCode::kSuccess) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  result_code = UpdateModel(update_model_req, fbb, device_meta);
  if (result_code != ResultCode::kSuccess) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    MS_LOG(DEBUG) << "Updating model failed.";
    return false;
  }
  std::string update_model_fl_id = update_model_req->fl_id()->str();
  IncreaseAcceptClientNum();
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());

  result_code = CountForAggregation(update_model_fl_id);
  if (result_code != ResultCode::kSuccess) {
    return false;
  }
  return true;
}

bool UpdateModelKernel::Reset() {
  MS_LOG(INFO) << "Update model kernel reset!";
  StopTimer();
  DistributedCountService::GetInstance().ResetCounter(name_);
  DistributedCountService::GetInstance().ResetCounter(kCountForAggregation);
  executor_->ResetAggregationStatus();
  DistributedMetadataStore::GetInstance().ResetMetadata(kCtxUpdateModelClientList);
  size_t &total_data_size = LocalMetaStore::GetInstance().mutable_value<size_t>(kCtxFedAvgTotalDataSize);
  total_data_size = 0;
  return true;
}

void UpdateModelKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {}

void UpdateModelKernel::RunAggregation() {
  auto is_last_iter_valid = Executor::GetInstance().RunAllWeightAggregation();
  auto curr_iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  if (is_last_iter_valid) {
    size_t total_data_size = LocalMetaStore::GetInstance().value<size_t>(kCtxFedAvgTotalDataSize);
    MS_LOG(INFO) << "Total data size for iteration " << curr_iter_num << " is " << total_data_size;
    if (ps::PSContext::instance()->resetter_round() == ps::ResetterRound::kUpdateModel &&
        ps::PSContext::instance()->encrypt_type() != ps::kPWEncryptType) {
      FinishIteration(is_last_iter_valid);
    }
  } else {
    std::string reason = "Weight aggregation failed, current iteration: " + std::to_string(curr_iter_num);
    MS_LOG(WARNING) << reason;
    FinishIteration(is_last_iter_valid, reason);
  }
}

ResultCode UpdateModelKernel::ReachThresholdForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                                           const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kFail);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->fl_id(), ResultCode::kFail);
  if (DistributedCountService::GetInstance().CountReachThreshold(name_, update_model_req->fl_id()->str())) {
    std::string reason = "Current amount for updateModel is enough. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

ResultCode UpdateModelKernel::VerifyUpdateModel(const schema::RequestUpdateModel *update_model_req,
                                                const std::shared_ptr<FBBuilder> &fbb, DeviceMeta *device_meta) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kFail);
  MS_ERROR_IF_NULL_W_RET_VAL(device_meta, ResultCode::kFail);

  std::string update_model_fl_id = update_model_req->fl_id()->str();
  size_t iteration = IntToSize(update_model_req->iteration());
  if (iteration != LocalMetaStore::GetInstance().curr_iter_num()) {
    auto next_req_time = LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp);
    std::string reason = "UpdateModel iteration number is invalid:" + std::to_string(iteration) +
                         ", current iteration:" + std::to_string(LocalMetaStore::GetInstance().curr_iter_num()) +
                         ", Retry later at time: " + std::to_string(next_req_time) + ", fl id is " + update_model_fl_id;
    BuildUpdateModelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(next_req_time));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }

  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(update_model_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed for fl id " + update_model_fl_id;
      BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed for fl id " + update_model_fl_id;
      BuildUpdateModelRsp(fbb, schema::ResponseCode_OutOfTime, reason, "");
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }
    MS_LOG(DEBUG) << "verify signature passed!";
  }

  std::unordered_map<std::string, size_t> feature_map;
  if (ps::PSContext::instance()->upload_compress_type() != kDiffSparseQuant) {
    auto upload_feature_map = update_model_req->feature_map();
    MS_ERROR_IF_NULL_W_RET_VAL(upload_feature_map, ResultCode::kFail);
    for (uint32_t i = 0; i < upload_feature_map->size(); i++) {
      const auto &item = upload_feature_map->Get(i);
      MS_ERROR_IF_NULL_W_RET_VAL(item, ResultCode::kFail);
      MS_ERROR_IF_NULL_W_RET_VAL(item->weight_fullname(), ResultCode::kFail);
      MS_ERROR_IF_NULL_W_RET_VAL(item->data(), ResultCode::kFail);

      std::string weight_full_name = item->weight_fullname()->str();
      size_t weight_size = item->data()->size() * sizeof(float);
      feature_map[weight_full_name] = weight_size;
    }
  }

  bool verifyFeatureMapIsSuccess;
  if (ps::PSContext::instance()->encrypt_type() == ps::kDSEncryptType && update_model_req->sign() != 0) {
    MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->index_array(), ResultCode::kFail);
    verifyFeatureMapIsSuccess = VerifySignDSFeatureMap(feature_map, update_model_req);
  } else if (ps::PSContext::instance()->upload_compress_type() == kDiffSparseQuant) {
    verifyFeatureMapIsSuccess = VerifyUploadCompressFeatureMap(update_model_req);
  } else {
    verifyFeatureMapIsSuccess = LocalMetaStore::GetInstance().verifyAggregationFeatureMap(feature_map);
  }
  if (!verifyFeatureMapIsSuccess) {
    auto next_req_time = LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp);
    std::string reason = "Verify model feature map failed, retry later at time: " + std::to_string(next_req_time);
    BuildUpdateModelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(next_req_time));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }

  MS_LOG(DEBUG) << "UpdateModel for fl id " << update_model_fl_id;

  bool found = DistributedMetadataStore::GetInstance().GetOneDeviceMeta(update_model_fl_id, device_meta);
  if (!found) {
    std::string reason = "devices_meta for " + update_model_fl_id + " is not set. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  if (ps::PSContext::instance()->encrypt_type() == ps::kPWEncryptType) {
    std::vector<std::string> get_secrets_clients;
#ifdef ENABLE_ARMOUR
    luojianet_ms::armour::CipherMetaStorage cipher_meta_storage;
    cipher_meta_storage.GetClientListFromServer(fl::server::kCtxGetSecretsClientList, &get_secrets_clients);
#endif
    if (find(get_secrets_clients.begin(), get_secrets_clients.end(), update_model_fl_id) ==
        get_secrets_clients.end()) {  // the client not in get_secrets_clients
      std::string reason = "fl_id: " + update_model_fl_id + " is not in get_secrets_clients. Please retry later.";
      BuildUpdateModelRsp(
        fbb, schema::ResponseCode_OutOfTime, reason,
        std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }
  }
  return ResultCode::kSuccess;
}

bool UpdateModelKernel::VerifySignDSFeatureMap(const std::unordered_map<std::string, size_t> &model,
                                               const schema::RequestUpdateModel *update_model_req) {
  auto &aggregation_feature_map_ = LocalMetaStore::GetInstance().aggregation_feature_map();
  if (model.size() > aggregation_feature_map_.size()) {
    return false;
  }
  auto index_array = update_model_req->index_array();
  size_t index_array_size = index_array->size();
  size_t array_size_upper = 100;
  if (index_array_size == 0 || index_array_size > array_size_upper) {
    return false;
  }
  for (const auto &weight : model) {
    std::string weight_name = weight.first;
    if (aggregation_feature_map_.count(weight_name) == 0) {
      return false;
    }
  }
  return true;
}

bool UpdateModelKernel::VerifyUploadCompressFeatureMap(const schema::RequestUpdateModel *update_model_req) {
  auto &aggregation_feature_map_ = LocalMetaStore::GetInstance().aggregation_feature_map();
  auto upload_sparse_rate = update_model_req->upload_sparse_rate();
  if (upload_sparse_rate != ps::PSContext::instance()->upload_sparse_rate()) {
    MS_LOG(WARNING) << "The upload_sparse_rate must be equal to the setting in context.";
    return false;
  }
  auto fbs_name_vec = update_model_req->name_vec();
  if (fbs_name_vec == nullptr) {
    MS_LOG(WARNING) << "The name_vec is null.";
    return false;
  }
  if (fbs_name_vec->size() == 0) {
    MS_LOG(WARNING) << "The size of name_vec must be larger than 0.";
    return false;
  }
  if (fbs_name_vec->size() > aggregation_feature_map_.size()) {
    MS_LOG(WARNING) << "The size of name_vec must be smaller than model in server.";
    return false;
  }
  for (size_t i = 0; i < fbs_name_vec->size(); ++i) {
    std::string name = fbs_name_vec->Get(i)->str();
    if (aggregation_feature_map_.count(name) == 0) {
      MS_LOG(WARNING) << "The upload name: " << name << " is not in model in server.";
      return false;
    }
  }
  auto fbs_compress_feature_map = update_model_req->compress_feature_map();
  if (fbs_compress_feature_map == nullptr) {
    MS_LOG(WARNING) << "The upload compress feature map is null.";
    return false;
  }
  if (fbs_compress_feature_map->size() == 0) {
    MS_LOG(WARNING) << "The upload compress feature map is empty.";
    return false;
  }
  return true;
}

ResultCode UpdateModelKernel::UpdateModel(const schema::RequestUpdateModel *update_model_req,
                                          const std::shared_ptr<FBBuilder> &fbb, const DeviceMeta &device_meta) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kFail);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->fl_id(), ResultCode::kFail);

  std::string update_model_fl_id = update_model_req->fl_id()->str();
  size_t data_size = device_meta.data_size();

  std::map<std::string, std::vector<float>> weight_map;
  std::map<std::string, UploadData> feature_map;
  if (ps::PSContext::instance()->encrypt_type() == ps::kDSEncryptType) {
    feature_map = ParseSignDSFeatureMap(update_model_req, data_size, &weight_map);
  } else if (ps::PSContext::instance()->upload_compress_type() == kDiffSparseQuant) {
    feature_map = ParseUploadCompressFeatureMap(update_model_req, data_size, &weight_map);
  } else {
    feature_map = ParseFeatureMap(update_model_req);
  }

  if (feature_map.empty()) {
    std::string reason = "Feature map is empty for fl id " + update_model_fl_id;
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }

  for (auto weight : feature_map) {
    weight.second[kNewDataSize].addr = &data_size;
    weight.second[kNewDataSize].size = sizeof(size_t);
    if (!executor_->HandleModelUpdate(weight.first, weight.second)) {
      std::string reason = "Updating weight " + weight.first + " failed for fl id " + update_model_fl_id;
      BuildUpdateModelRsp(
        fbb, schema::ResponseCode_OutOfTime, reason,
        std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }
  }

  FLId fl_id;
  fl_id.set_fl_id(update_model_fl_id);
  PBMetadata comm_value;
  *comm_value.mutable_fl_id() = fl_id;
  if (!DistributedMetadataStore::GetInstance().UpdateMetadata(kCtxUpdateModelClientList, comm_value)) {
    std::string reason = "Updating metadata of UpdateModelClientList failed for fl id " + update_model_fl_id;
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  UpdateClientUploadLoss(update_model_req->upload_loss());
  BuildUpdateModelRsp(fbb, schema::ResponseCode_SUCCEED, "success not ready",
                      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
  return ResultCode::kSuccess;
}

std::map<std::string, UploadData> UpdateModelKernel::ParseFeatureMap(
  const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, {});
  std::map<std::string, UploadData> feature_map;
  auto fbs_feature_map = update_model_req->feature_map();
  MS_ERROR_IF_NULL_W_RET_VAL(fbs_feature_map, feature_map);
  for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
    std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
    float *weight_data = const_cast<float *>(fbs_feature_map->Get(i)->data()->data());
    size_t weight_size = fbs_feature_map->Get(i)->data()->size() * sizeof(float);
    UploadData upload_data;
    upload_data[kNewWeight].addr = weight_data;
    upload_data[kNewWeight].size = weight_size;
    feature_map[weight_full_name] = upload_data;
  }
  return feature_map;
}

std::map<std::string, UploadData> UpdateModelKernel::ParseSignDSFeatureMap(
  const schema::RequestUpdateModel *update_model_req, size_t data_size,
  std::map<std::string, std::vector<float>> *weight_map) {
  auto fbs_feature_map = update_model_req->feature_map();
  std::map<std::string, UploadData> feature_map;
  auto sign = update_model_req->sign();
  if (sign == 0) {
    feature_map = ParseFeatureMap(update_model_req);
    return feature_map;
  }

  const auto &iter_to_model = ModelStore::GetInstance().iteration_to_model();
  size_t latest_iter_num = iter_to_model.rbegin()->first;
  std::map<std::string, AddressPtr> feature_maps_store = ModelStore::GetInstance().GetModelByIterNum(latest_iter_num);
  auto index_array = update_model_req->index_array();
  size_t index_store = 0;
  size_t index_array_j = 0;
  float signds_grad = sign * ps::PSContext::instance()->sign_global_lr();
  for (size_t i = 0; i < fbs_feature_map->size(); i++) {
    std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
    AddressPtr iter_feature_map_data_ptr = feature_maps_store[weight_full_name];
    size_t iter_feature_num = iter_feature_map_data_ptr->size / sizeof(float);
    auto &weight_item = (*weight_map)[weight_full_name];
    weight_item.resize(iter_feature_num);
    float *iter_feature_map_data = reinterpret_cast<float *>(iter_feature_map_data_ptr->addr);
    for (size_t j = 0; j < iter_feature_num; j++) {
      float reconstruct_weight = iter_feature_map_data[j];
      if (index_array_j < index_array->size() && index_store == static_cast<size_t>(index_array->Get(index_array_j))) {
        reconstruct_weight += signds_grad;
        index_array_j++;
      }
      reconstruct_weight *= data_size;
      index_store++;
      weight_item[j] = reconstruct_weight;
    }
    size_t weight_size = iter_feature_num * sizeof(float);
    UploadData upload_data;
    upload_data[kNewWeight].addr = weight_item.data();
    upload_data[kNewWeight].size = weight_size;
    feature_map[weight_full_name] = upload_data;
  }
  return feature_map;
}

std::map<std::string, UploadData> UpdateModelKernel::ParseUploadCompressFeatureMap(
  const schema::RequestUpdateModel *update_model_req, size_t data_size,
  std::map<std::string, std::vector<float>> *weight_map) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, {});
  std::map<std::string, UploadData> feature_map;
  schema::CompressType upload_compress_type = update_model_req->upload_compress_type();
  upload_compress_type =
    luojianet_ms::fl::compression::DecodeExecutor::GetInstance().GetCompressType(upload_compress_type);
  MS_LOG(INFO) << "This schema upload compress type is: " << upload_compress_type;
  if (upload_compress_type != schema::CompressType_NO_COMPRESS) {
    MS_LOG(INFO) << "This upload compress type is DIFF_SPARSE_QUANT.";
    feature_map = DecodeFeatureMap(weight_map, update_model_req, upload_compress_type, data_size);
    return feature_map;
  }
  MS_LOG(INFO) << "This upload compress type is NO_COMPRESS.";
  // Some clients upload origin weights.
  auto fbs_feature_map = update_model_req->feature_map();
  MS_ERROR_IF_NULL_W_RET_VAL(fbs_feature_map, feature_map);
  for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
    std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
    float *weight_data = const_cast<float *>(fbs_feature_map->Get(i)->data()->data());
    size_t weight_size = fbs_feature_map->Get(i)->data()->size() * sizeof(float);
    UploadData upload_data;
    upload_data[kNewWeight].addr = weight_data;
    upload_data[kNewWeight].size = weight_size;
    feature_map[weight_full_name] = upload_data;
  }
  return feature_map;
}

std::map<std::string, UploadData> UpdateModelKernel::DecodeFeatureMap(
  std::map<std::string, std::vector<float>> *weight_map, const schema::RequestUpdateModel *update_model_req,
  schema::CompressType upload_compress_type, size_t data_size) {
  std::map<std::string, UploadData> feature_map;

  // Get and set decode hyper parameters.
  auto seed = update_model_req->iteration();
  MS_LOG(INFO) << "The seed for compression is: " << seed;
  auto upload_sparse_rate = update_model_req->upload_sparse_rate();
  MS_LOG(INFO) << "The upload_sparse_rate for compression is: " << upload_sparse_rate;
  // Get name vector.
  auto fbs_name_vec = update_model_req->name_vec();
  std::vector<std::string> name_vec;
  for (size_t i = 0; i < fbs_name_vec->size(); ++i) {
    name_vec.emplace_back(fbs_name_vec->Get(i)->str());
  }

  // Parameter process for decode.
  auto fbs_compress_feature_map = update_model_req->compress_feature_map();
  std::vector<luojianet_ms::fl::compression::CompressFeatureMap> compress_feature_maps;
  for (size_t i = 0; i < fbs_compress_feature_map->size(); ++i) {
    luojianet_ms::fl::compression::CompressFeatureMap compress_feature_map;
    int8_t *compress_weight_data = const_cast<int8_t *>(fbs_compress_feature_map->Get(i)->compress_data()->data());
    size_t compress_weight_size = fbs_compress_feature_map->Get(i)->compress_data()->size();
    MS_LOG(INFO) << "The compress weight size: " << compress_weight_size;
    for (size_t j = 0; j < compress_weight_size; ++j) {
      compress_feature_map.compress_data.emplace_back(compress_weight_data[j]);
    }
    compress_feature_map.min_val = fbs_compress_feature_map->Get(i)->min_val();
    compress_feature_map.max_val = fbs_compress_feature_map->Get(i)->max_val();
    MS_LOG(INFO) << "Min value: " << compress_feature_map.min_val;
    MS_LOG(INFO) << "Max value: " << compress_feature_map.max_val;
    compress_feature_maps.emplace_back(compress_feature_map);
  }

  // Decode.
  bool status = luojianet_ms::fl::compression::DecodeExecutor::GetInstance().Decode(
    weight_map, compress_feature_maps, upload_compress_type, upload_sparse_rate, seed, name_vec, data_size);
  if (status) {
    for (size_t i = 0; i < name_vec.size(); ++i) {
      std::string weight_full_name = name_vec[i];
      size_t weight_size = (*weight_map)[weight_full_name].size() * sizeof(float);
      UploadData upload_data;
      upload_data[kNewWeight].addr = (*weight_map)[weight_full_name].data();
      upload_data[kNewWeight].size = weight_size;
      feature_map[weight_full_name] = upload_data;
    }
    return feature_map;
  }
  MS_LOG(WARNING) << "Decode failed!";
  return feature_map;
}

ResultCode UpdateModelKernel::CountForAggregation(const std::string &req_fl_id) {
  if (!DistributedCountService::GetInstance().Count(kCountForAggregation, req_fl_id)) {
    MS_LOG(ERROR) << "Counting for aggregation failed for fl id " << req_fl_id;
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

ResultCode UpdateModelKernel::CountForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                                  const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(fbb, ResultCode::kFail);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kFail);
  if (!DistributedCountService::GetInstance().Count(name_, update_model_req->fl_id()->str())) {
    std::string reason = "Counting for update model request failed for fl id " + update_model_req->fl_id()->str() +
                         ", Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

sigVerifyResult UpdateModelKernel::VerifySignature(const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = update_model_req->fl_id()->str();
  std::string timestamp = update_model_req->timestamp()->str();
  int iteration = update_model_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = update_model_req->signature();
  std::vector<unsigned char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(WARNING) << "signature in client_list_sign_req is nullptr";
    return sigVerifyResult::FAILED;
  }
  signature.assign(fbs_signature->begin(), fbs_signature->end());
  std::map<std::string, std::string> key_attestations;
  const fl::PBMetadata &key_attestations_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(kCtxClientKeyAttestation);
  const fl::KeyAttestation &key_attestation_pb = key_attestations_pb_out.key_attestation();
  auto iter = key_attestation_pb.key_attestations().begin();
  for (; iter != key_attestation_pb.key_attestations().end(); ++iter) {
    (void)key_attestations.emplace(std::pair<std::string, std::string>(iter->first, iter->second));
  }
  if (key_attestations.find(fl_id) == key_attestations.end()) {
    MS_LOG(WARNING) << "can not find key attestation for fl_id: " << fl_id;
    return sigVerifyResult::TIMEOUT;
  }

  std::vector<unsigned char> src_data;
  (void)src_data.insert(src_data.end(), timestamp.begin(), timestamp.end());
  (void)src_data.insert(src_data.end(), iter_str.begin(), iter_str.end());
  auto certVerify = luojianet_ms::ps::server::CertVerify::GetInstance();
  unsigned char srcDataHash[SHA256_DIGEST_LENGTH];
  certVerify.sha256Hash(src_data.data(), SizeToInt(src_data.size()), srcDataHash, SHA256_DIGEST_LENGTH);
  if (!certVerify.verifyRSAKey(key_attestations[fl_id], srcDataHash, signature.data(), SHA256_DIGEST_LENGTH)) {
    return sigVerifyResult::FAILED;
  }
  if (!certVerify.verifyTimeStamp(fl_id, timestamp)) {
    return sigVerifyResult::TIMEOUT;
  }
  MS_LOG(INFO) << "verify signature for fl_id: " << fl_id << " success.";
  return sigVerifyResult::PASSED;
}

void UpdateModelKernel::BuildUpdateModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                            const std::string &reason, const std::string &next_req_time) {
  if (fbb == nullptr) {
    MS_LOG(WARNING) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);

  schema::ResponseUpdateModelBuilder rsp_update_model_builder(*(fbb.get()));
  rsp_update_model_builder.add_retcode(static_cast<int>(retcode));
  rsp_update_model_builder.add_reason(fbs_reason);
  rsp_update_model_builder.add_next_req_time(fbs_next_req_time);
  auto rsp_update_model = rsp_update_model_builder.Finish();
  fbb->Finish(rsp_update_model);
  return;
}

REG_ROUND_KERNEL(updateModel, UpdateModelKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace luojianet_ms
