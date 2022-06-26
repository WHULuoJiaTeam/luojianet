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

#include "fl/server/model_store.h"
#include <map>
#include <string>
#include <memory>
#include "fl/server/executor.h"
#include "pipeline/jit/parse/parse.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace fl {
namespace server {
void ModelStore::Initialize(uint32_t rank_id, uint32_t max_count) {
  if (!Executor::GetInstance().initialized()) {
    MS_LOG(EXCEPTION) << "Server's executor must be initialized before model storage.";
    return;
  }
  rank_id_ = rank_id;
  max_model_count_ = max_count;
  initial_model_ = AssignNewModelMemory();
  iteration_to_model_[kInitIterationNum] = initial_model_;
  std::map<std::string, AddressPtr> model = Executor::GetInstance().GetModel();
  for (const auto &item : mindspore::fl::compression::kCompressTypeMap) {
    iteration_to_compress_model_[kInitIterationNum][item.first] = AssignNewCompressModelMemory(item.first, model);
  }
  model_size_ = ComputeModelSize();
  MS_LOG(INFO) << "Model store checkpoint dir is: " << ps::PSContext::instance()->checkpoint_dir();
}

void ModelStore::StoreModelByIterNum(size_t iteration, const std::map<std::string, AddressPtr> &new_model) {
  std::unique_lock<std::mutex> lock(model_mtx_);
  if (iteration_to_model_.count(iteration) != 0) {
    MS_LOG(WARNING) << "Model for iteration " << iteration << " is already stored";
    return;
  }
  if (new_model.empty()) {
    MS_LOG(ERROR) << "Model feature map is empty.";
    return;
  }

  std::shared_ptr<MemoryRegister> memory_register = nullptr;
  if (iteration_to_model_.size() < max_model_count_) {
    // If iteration_to_model_.size() is not max_model_count_, need to assign new memory for the model.
    memory_register = AssignNewModelMemory();
    MS_ERROR_IF_NULL_WO_RET_VAL(memory_register);
    iteration_to_model_[iteration] = memory_register;
  } else {
    // If iteration_to_model_ size is already max_model_count_, we need to replace earliest model with the newest model.
    memory_register = iteration_to_model_.begin()->second;
    MS_ERROR_IF_NULL_WO_RET_VAL(memory_register);
    (void)iteration_to_model_.erase(iteration_to_model_.begin());
  }

  // Copy new model data to the the stored model.
  auto &stored_model = memory_register->addresses();
  for (const auto &weight : new_model) {
    const std::string &weight_name = weight.first;
    if (stored_model.count(weight_name) == 0) {
      MS_LOG(ERROR) << "The stored model has no weight " << weight_name;
      continue;
    }

    MS_ERROR_IF_NULL_WO_RET_VAL(stored_model[weight_name]);
    MS_ERROR_IF_NULL_WO_RET_VAL(stored_model[weight_name]->addr);
    MS_ERROR_IF_NULL_WO_RET_VAL(weight.second);
    MS_ERROR_IF_NULL_WO_RET_VAL(weight.second->addr);
    void *dst_addr = stored_model[weight_name]->addr;
    size_t dst_size = stored_model[weight_name]->size;
    void *src_addr = weight.second->addr;
    size_t src_size = weight.second->size;
    int ret = memcpy_s(dst_addr, dst_size, src_addr, src_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return;
    }
  }
  iteration_to_model_[iteration] = memory_register;
  OnIterationUpdate();
  SaveCheckpoint(iteration, new_model);
  return;
}

std::map<std::string, AddressPtr> ModelStore::GetModelByIterNum(size_t iteration) {
  std::unique_lock<std::mutex> lock(model_mtx_);
  std::map<std::string, AddressPtr> model = {};
  if (iteration_to_model_.count(iteration) == 0) {
    MS_LOG(WARNING) << "Model for iteration " << iteration << " is not stored. Return latest model";
    return model;
  }
  model = iteration_to_model_[iteration]->addresses();
  return model;
}

std::map<std::string, AddressPtr> ModelStore::GetCompressModelByIterNum(size_t iteration,
                                                                        schema::CompressType compressType) {
  std::unique_lock<std::mutex> lock(model_mtx_);
  std::map<std::string, AddressPtr> compressModel = {};
  if (iteration_to_compress_model_.count(iteration) == 0) {
    MS_LOG(ERROR) << "Compress Model for iteration " << iteration << " is not stored.";
    return compressModel;
  }
  std::map<schema::CompressType, std::shared_ptr<MemoryRegister>> compress_model_map =
    iteration_to_compress_model_[iteration];
  if (compress_model_map.count(compressType) == 0) {
    MS_LOG(ERROR) << "Compress Model for compress type " << compressType << " is not stored.";
    return compressModel;
  }
  compressModel = iteration_to_compress_model_[iteration][compressType]->addresses();
  return compressModel;
}

void ModelStore::Reset() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  initial_model_ = iteration_to_model_.rbegin()->second;
  iteration_to_model_.clear();
  iteration_to_model_[kInitIterationNum] = initial_model_;
  OnIterationUpdate();
}

const std::map<size_t, std::shared_ptr<MemoryRegister>> &ModelStore::iteration_to_model() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  return iteration_to_model_;
}

const std::map<size_t, CompressTypeMap> &ModelStore::iteration_to_compress_model() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  return iteration_to_compress_model_;
}

size_t ModelStore::model_size() const { return model_size_; }

std::shared_ptr<MemoryRegister> ModelStore::AssignNewModelMemory() {
  std::map<std::string, AddressPtr> model = Executor::GetInstance().GetModel();
  if (model.empty()) {
    MS_LOG(WARNING) << "Model feature map is empty.";
    return std::make_shared<MemoryRegister>();
  }

  // Assign new memory for the model.
  std::shared_ptr<MemoryRegister> memory_register = std::make_shared<MemoryRegister>();
  MS_ERROR_IF_NULL_W_RET_VAL(memory_register, nullptr);
  for (const auto &weight : model) {
    const std::string weight_name = weight.first;
    size_t weight_size = weight.second->size;
    auto weight_data = std::make_unique<char[]>(weight_size);
    MS_ERROR_IF_NULL_W_RET_VAL(weight_data, nullptr);
    MS_ERROR_IF_NULL_W_RET_VAL(weight.second, nullptr);
    MS_ERROR_IF_NULL_W_RET_VAL(weight.second->addr, nullptr);

    auto src_data_size = weight_size;
    auto dst_data_size = weight_size;
    int ret = memcpy_s(weight_data.get(), dst_data_size, weight.second->addr, src_data_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return nullptr;
    }
    memory_register->RegisterArray(weight_name, &weight_data, weight_size);
  }
  return memory_register;
}

std::shared_ptr<MemoryRegister> ModelStore::AssignNewCompressModelMemory(
  schema::CompressType compressType, const std::map<std::string, AddressPtr> &model) {
  if (model.empty()) {
    MS_LOG(EXCEPTION) << "Model feature map is empty.";
    return nullptr;
  }
  std::map<string, std::vector<float>> feature_maps;
  for (auto &feature_map : model) {
    auto weight_fullname = feature_map.first;
    auto weight_data = reinterpret_cast<float *>(feature_map.second->addr);
    std::vector<float> weight_data_vector{weight_data, weight_data + feature_map.second->size / sizeof(float)};
    feature_maps[weight_fullname] = weight_data_vector;
  }

  std::map<std::string, mindspore::fl::compression::CompressWeight> compressWeights;
  bool status = mindspore::fl::compression::CompressExecutor::GetInstance().construct_compress_weight(
    &compressWeights, feature_maps, compressType);
  if (!status) {
    MS_LOG(ERROR) << "Encode failed!";
    return nullptr;
  }

  // Assign new memory for the compress model.
  std::shared_ptr<MemoryRegister> memory_register = std::make_shared<MemoryRegister>();
  MS_ERROR_IF_NULL_W_RET_VAL(memory_register, nullptr);
  MS_LOG(INFO) << "Register compressWeight for compressType: " << schema::EnumNameCompressType(compressType);

  for (const auto &compressWeight : compressWeights) {
    if (compressType == schema::CompressType_QUANT) {
      std::string compress_weight_name = compressWeight.first;
      std::string min_val_name = compress_weight_name + "." + kMinVal;
      std::string max_val_name = compress_weight_name + "." + kMaxVal;
      size_t compress_weight_size = compressWeight.second.compress_data_len * sizeof(int8_t);
      auto compress_weight_data = std::make_unique<char[]>(compress_weight_size);
      auto src_data_size = compress_weight_size;
      auto dst_data_size = compress_weight_size;
      int ret =
        memcpy_s(compress_weight_data.get(), dst_data_size, compressWeight.second.compress_data.data(), src_data_size);
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return nullptr;
      }
      memory_register->RegisterArray(compress_weight_name, &compress_weight_data, compress_weight_size);
      size_t float_size = 1;
      auto min_val_ptr = std::make_unique<float>(compressWeight.second.min_val);
      auto max_val_ptr = std::make_unique<float>(compressWeight.second.max_val);

      memory_register->RegisterParameter(min_val_name, &min_val_ptr, float_size);
      memory_register->RegisterParameter(max_val_name, &max_val_ptr, float_size);
    }
  }
  return memory_register;
}

void ModelStore::StoreCompressModelByIterNum(size_t iteration, const std::map<std::string, AddressPtr> &new_model) {
  std::unique_lock<std::mutex> lock(model_mtx_);
  if (iteration_to_compress_model_.count(iteration) != 0) {
    MS_LOG(WARNING) << "Compress Model for iteration " << iteration << " is already stored";
    return;
  }
  if (new_model.empty()) {
    MS_LOG(ERROR) << "Compress Model feature map is empty.";
    return;
  }

  iteration_to_compress_model_[iteration] = {};
  if (iteration_to_compress_model_.size() >= max_model_count_) {
    auto compress_model_map = iteration_to_compress_model_.begin()->second;
    compress_model_map.clear();
    (void)iteration_to_compress_model_.erase(iteration_to_compress_model_.begin());
  }

  for (const auto &item : mindspore::fl::compression::kCompressTypeMap) {
    auto memory_register = AssignNewCompressModelMemory(item.first, new_model);
    MS_ERROR_IF_NULL_WO_RET_VAL(memory_register);
    iteration_to_compress_model_[iteration][item.first] = memory_register;
  }
  return;
}

size_t ModelStore::ComputeModelSize() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  if (iteration_to_model_.empty()) {
    MS_LOG(EXCEPTION) << "Calculating model size failed: model for iteration 0 is not stored yet. ";
    return 0;
  }

  const auto &model = iteration_to_model_[kInitIterationNum];
  MS_EXCEPTION_IF_NULL(model);
  size_t model_size = std::accumulate(model->addresses().begin(), model->addresses().end(), static_cast<size_t>(0),
                                      [](size_t s, const auto &weight) { return s + weight.second->size; });
  MS_LOG(INFO) << "Model size in byte is " << model_size;
  return model_size;
}

void ModelStore::RelModelResponseCache(const void *data, size_t datalen, void *extra) {
  auto &instance = GetInstance();
  std::unique_lock<std::mutex> lock(instance.model_response_cache_lock_);
  auto it =
    std::find_if(instance.model_response_cache_.begin(), instance.model_response_cache_.end(),
                 [data](const HttpResponseModelCache &item) { return item.cache && item.cache->data() == data; });
  if (it == instance.model_response_cache_.end()) {
    MS_LOG(WARNING) << "Model response cache has been releaed";
    return;
  }
  if (it->reference_count > 0) {
    it->reference_count -= 1;
    instance.total_sub_reference_count++;
  }
}

std::shared_ptr<std::vector<uint8_t>> ModelStore::GetModelResponseCache(const string &round_name,
                                                                        size_t cur_iteration_num,
                                                                        size_t model_iteration_num,
                                                                        const std::string &compress_type) {
  std::unique_lock<std::mutex> lock(model_response_cache_lock_);
  auto it = std::find_if(
    model_response_cache_.begin(), model_response_cache_.end(),
    [&round_name, cur_iteration_num, model_iteration_num, &compress_type](const HttpResponseModelCache &item) {
      return item.round_name == round_name && item.cur_iteration_num == cur_iteration_num &&
             item.model_iteration_num == model_iteration_num && item.compress_type == compress_type;
    });
  if (it == model_response_cache_.end()) {
    return nullptr;
  }
  it->reference_count += 1;
  total_add_reference_count += 1;
  return it->cache;
}

std::shared_ptr<std::vector<uint8_t>> ModelStore::StoreModelResponseCache(const string &round_name,
                                                                          size_t cur_iteration_num,
                                                                          size_t model_iteration_num,
                                                                          const std::string &compress_type,
                                                                          const void *data, size_t datalen) {
  std::unique_lock<std::mutex> lock(model_response_cache_lock_);
  auto it = std::find_if(
    model_response_cache_.begin(), model_response_cache_.end(),
    [&round_name, cur_iteration_num, model_iteration_num, &compress_type](const HttpResponseModelCache &item) {
      return item.round_name == round_name && item.cur_iteration_num == cur_iteration_num &&
             item.model_iteration_num == model_iteration_num && item.compress_type == compress_type;
    });
  if (it != model_response_cache_.end()) {
    it->reference_count += 1;
    total_add_reference_count += 1;
    return it->cache;
  }
  auto cache = std::make_shared<std::vector<uint8_t>>(datalen);
  if (cache == nullptr) {
    MS_LOG(ERROR) << "Malloc data of size " << datalen << " failed";
    return nullptr;
  }
  auto ret = memcpy_s(cache->data(), cache->size(), data, datalen);
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s  error, errorno(" << ret << ")";
    return nullptr;
  }
  HttpResponseModelCache item;
  item.round_name = round_name;
  item.cur_iteration_num = cur_iteration_num;
  item.model_iteration_num = model_iteration_num;
  item.compress_type = compress_type;
  item.cache = cache;
  item.reference_count = 1;
  total_add_reference_count += 1;
  model_response_cache_.push_back(item);
  return cache;
}

void ModelStore::OnIterationUpdate() {
  std::unique_lock<std::mutex> lock(model_response_cache_lock_);
  for (auto it = model_response_cache_.begin(); it != model_response_cache_.end();) {
    if (it->reference_count == 0) {
      it->cache = nullptr;
      it = model_response_cache_.erase(it);
    } else {
      ++it;
    }
  }
  MS_LOG(INFO) << "Current model cache number: " << model_response_cache_.size()
               << ", total add and sub reference count: " << total_add_reference_count << ", "
               << total_sub_reference_count;
}

void ModelStore::SaveCheckpoint(size_t iteration, const std::map<std::string, AddressPtr> &model) {
  if (rank_id_ != kLeaderServerRank) {
    MS_LOG(INFO) << "Only leader server will save the weight.";
    return;
  }
  std::unordered_map<std::string, Feature> &aggregation_feature_map =
    LocalMetaStore::GetInstance().aggregation_feature_map();

  namespace python_adapter = mindspore::python_adapter;
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_SERIALIZE_MODULE);

  py::dict dict_data = py::dict();
  for (const auto &weight : model) {
    std::string weight_fullname = weight.first;
    float *weight_data = reinterpret_cast<float *>(weight.second->addr);
    size_t weight_data_size = weight.second->size / sizeof(float);
    Feature aggregation_feature = aggregation_feature_map[weight_fullname];

    std::vector<float> weight_data_vec(weight_data, weight_data + weight_data_size);

    py::list data_list;
    data_list.append(aggregation_feature.weight_type);
    data_list.append(aggregation_feature.weight_shape);
    data_list.append(weight_data_vec);
    data_list.append(weight_data_size);
    dict_data[py::str(weight_fullname)] = data_list;
  }

  std::string checkpoint_dir = ps::PSContext::instance()->checkpoint_dir();
  std::string fl_name = ps::PSContext::instance()->fl_name();

  python_adapter::CallPyModFn(mod, PYTHON_MOD_SAFE_WEIGHT, py::str(checkpoint_dir), py::str(fl_name),
                              py::str(std::to_string(iteration)), dict_data);
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
