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

#include "ps/parameter_server.h"
#include <algorithm>
#include <thread>
#include <set>

#include "utils/file_utils.h"

namespace mindspore {
namespace ps {
static const uint32_t kMaxThreadNum = 16;
static const uint32_t kCPUCoreNum = std::thread::hardware_concurrency();

ParameterServer &ParameterServer::GetInstance() {
  static ParameterServer instance{};
  return instance;
}

void ParameterServer::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "PServer starts connecting to scheduler and workers...";
  server_node_ = std::make_shared<core::PSServerNode>();

  MS_LOG(INFO) << "PServer connected successfully.";
  if (!PSContext::instance()->is_server()) {
    MS_LOG(INFO) << "This is not the Server node.";
    return;
  }
  Init(func_graph);
  server_node_->Start();

  if (EnableRecovery()) {
    MS_EXCEPTION_IF_NULL(recover_handler_);
    recover_handler_->Init();
    recover_handler_->Recover();
    finish_recovery_ = true;
  }

  PSContext::instance()->SetPSRankId(server_node_->rank_id());
  thread_->join();
  SyncEmbeddingTables();
  MS_LOG(INFO) << "PServer finished updating models, starts finalizing...";
  server_node_->Finish();
  if (!server_node_->Stop()) {
    MS_LOG(WARNING) << "Parameter server stop failed.";
  }
  MS_LOG(INFO) << "PServer finalized successfully.";
}

bool ParameterServer::Init(const FuncGraphPtr &func_graph) {
  pserver_num_ = std::strtol(mindspore::common::GetEnv(kEnvPServerNum).c_str(), nullptr, kBase);
  worker_num_ = std::strtol(mindspore::common::GetEnv(kEnvWorkerNum).c_str(), nullptr, kBase);
  func_graph_ = func_graph;
  handler_.reset(new ServerHandler(this));
  handler_->Init();

  recover_handler_ = std::make_unique<RecoverHandler>(this);

  InitOptimInfoBuilders();
  server_node_->set_handler(*handler_);
  server_node_->RegisterEventCallback(core::ClusterEvent::SCHEDULER_TIMEOUT, [this]() {
    MS_LOG(ERROR) << "Trigger timeout event: SCHEDULER_TIMEOUT begin to exit the system!";
    this->Finalize();
  });
  server_node_->RegisterEventCallback(core::ClusterEvent::NODE_TIMEOUT, [this]() {
    MS_LOG(ERROR) << "Trigger timeout event: NODE_TIMEOUT begin to exit the system!";
    this->Finalize();
  });
  server_node_->RegisterEventCallback(core::ClusterEvent::ON_BEGIN_PERSIST, [this]() { this->PersistParameters(); });
  thread_.reset(new std::thread(&ParameterServer::UpdateWeights, this));
  GetEmbeddingTableParamPtr();
  return true;
}

void ParameterServer::InitOptimInfoBuilders() {
  std::shared_ptr<OptimizerInfoBuilder> momentum_info_builder = std::make_shared<MomentumOptimInfoBuilder>(worker_num_);
  std::shared_ptr<OptimizerInfoBuilder> sparse_adam_info_builder =
    std::make_shared<SparseAdamOptimInfoBuilder>(worker_num_);
  std::shared_ptr<OptimizerInfoBuilder> sparse_ftrl_info_builder =
    std::make_shared<SparseFtrlOptimInfoBuilder>(worker_num_);
  optim_info_builders_[kApplyMomentum] = momentum_info_builder;
  optim_info_builders_[kSparseAdam] = sparse_adam_info_builder;
  optim_info_builders_[kSparseFtrl] = sparse_ftrl_info_builder;
}

void ParameterServer::InitWeightKeyToOptims(const Key &key, const int64_t &optim_id) {
  if (weight_key_to_optims_.count(key) > 0 || Util::optimizer_name(optim_id) == "") {
    return;
  }
  weight_key_to_optims_[key] = Util::optimizer_name(optim_id);
  weight_key_to_optim_op_[key] = Util::optimizer_node_name(optim_id);
  MS_LOG(INFO) << "Initializing optimizer id for key:" << key << ", optimizer name:" << weight_key_to_optims_[key]
               << ", optimizer op name:" << weight_key_to_optim_op_[key];
}

void ParameterServer::InitOptimInputsShape(const Keys &keys, const Values &values, const Lengths &lengths) {
  InputsShapePtr inputs_shape = std::make_shared<InputsShape>();
  MS_EXCEPTION_IF_NULL(inputs_shape);
  InputsShapePtr original_inputs_shape = std::make_shared<InputsShape>();
  MS_EXCEPTION_IF_NULL(original_inputs_shape);
  size_t val_idx = 0;
  const Key &key = keys[0];
  MS_LOG(INFO) << "Initializing optimizer inputs shape for key:" << key;
  if (optim_inputs_shape_.count(key) == 0) {
    original_optim_inputs_shape_[key] = original_inputs_shape;
    optim_inputs_shape_[key] = inputs_shape;
  }
  for (size_t i = 0; i < keys.size(); i++) {
    auto shape = std::make_shared<std::vector<size_t>>();
    MS_EXCEPTION_IF_NULL(shape);
    auto original_shape = std::make_shared<std::vector<size_t>>();
    MS_EXCEPTION_IF_NULL(original_shape);
    inputs_shape->push_back(shape);
    original_inputs_shape->push_back(original_shape);

    for (int64_t j = 0; j < lengths[i]; j++) {
      shape->push_back(values[val_idx]);
      original_shape->push_back(values[val_idx++]);
    }
  }
  if (weight_key_to_optims_.count(key) > 0) {
    const std::string &optim_name = weight_key_to_optims_[key];
    const std::string &optim_op_name = weight_key_to_optim_op_[key];
    if (optimizers_.count(key) == 0 && optim_inputs_shape_.count(key) > 0) {
      const CNodePtr cnode = GetCNode(optim_op_name);
      MS_EXCEPTION_IF_NULL(cnode);
      if (optim_name == kSparseAdam) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyAdamPSKernelMod>(server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseLazyAdam) {
        std::shared_ptr<PServerKernel> optimizer = std::make_shared<kernel::ps::SparseApplyLazyAdamPSKernelMod>(
          server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kApplyMomentum) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::ApplyMomentumPSKernelMod>(server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseFtrl) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyFtrlPSKernelMod>(server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      }
    }
  }
}

void ParameterServer::InitWeight(const Key &key, const WeightPtr &weight) {
  MS_EXCEPTION_IF_NULL(weight);
  if ((weights_.count(key) == 0) || (is_embedding_[key] && weights_.count(key) != 0)) {
    MS_LOG(INFO) << "Initializing weight for key " << key << ", server rank " << server_node_->rank_id();
    weights_[key] = weight;
    tokens_[key] = 0;
    is_embedding_[key] = false;
  }
}

void ParameterServer::InitGrad(const Key &key, const GradPtr &grad) {
  MS_EXCEPTION_IF_NULL(grad);
  if (grads_.count(key) == 0) {
    grads_[key] = grad;
    grads_accum_counter_[key] = 0;
  }
}

namespace {
// Initialize accumulation by multithreading parallelism.
void InitAccumParallel(float init_value, size_t total_len, float *embedding_data) {
  MS_EXCEPTION_IF_NULL(embedding_data);
  auto init_task = [](float value, size_t task_len, float *data) {
    for (size_t i = 0; i < task_len; i++) {
      data[i] = value;
    }
  };

  size_t thread_num = std::max(kMaxThreadNum, kCPUCoreNum);
  if (total_len <= thread_num) {
    thread_num = 1;
  }

  std::vector<std::thread> threads(thread_num);
  size_t task_offset = 0;

  for (size_t i = 0; i < thread_num; ++i) {
    // The value of thread_num is >= 1.
    size_t task_len = total_len / thread_num + (i < (total_len % thread_num) ? 1 : 0);
    threads[i] = std::thread(init_task, init_value, task_len, embedding_data + task_offset);
    task_offset += task_len;
  }

  for (size_t i = 0; i < thread_num; i++) {
    threads[i].join();
  }
}

void CopyTensorData(void *dest_ptr, size_t tensor_size, const void *src_ptr) {
  MS_EXCEPTION_IF_NULL(dest_ptr);
  MS_EXCEPTION_IF_NULL(src_ptr);
  char *dest = reinterpret_cast<char *>(dest_ptr);
  const char *src = reinterpret_cast<const char *>(src_ptr);

  // The security memcpy function 'memcpy_s' limits the value of the second parameter 'destMax' not to be greater than
  // SECUREC_MEM_MAX_LEN. If tensor size(buffer length) is greater than SECUREC_MEM_MAX_LEN, the tensor should be cut
  // into segments to copy.
  for (size_t offset = 0; offset < tensor_size; offset += SECUREC_MEM_MAX_LEN) {
    size_t copy_len = std::min(tensor_size - offset, SECUREC_MEM_MAX_LEN);
    size_t dest_len = copy_len;
    int ret = memcpy_s(dest + offset, dest_len, src + offset, copy_len);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Failed to memcpy tensor, errorno(" << ret << ")";
    }
  }
}
}  // namespace

void ParameterServer::PersistKernels(const Key &key,
                                     const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes,
                                     const ParamInitInfo &param_init_info) const {
  if (!EnableRecovery()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(shapes);
  MS_EXCEPTION_IF_NULL(recover_handler_);
  auto *config_storage = recover_handler_->config_storage();
  MS_EXCEPTION_IF_NULL(config_storage);
  std::vector<std::string> recover_funcs;
  if (config_storage->Exists(kRecoverFunc)) {
    recover_funcs = config_storage->GetValue<std::vector<std::string>>(kRecoverFunc);
  }
  std::string recover_embedding = kRecoverEmbedding;
  if (!std::any_of(recover_funcs.begin(), recover_funcs.end(),
                   [&](const std::string &func_name) { return func_name == recover_embedding; })) {
    recover_funcs.push_back(recover_embedding);
    config_storage->PutValue(kRecoverFunc, recover_funcs);
  }

  // Persist key.
  std::vector<Key> keys;
  if (config_storage->Exists(kKeys)) {
    keys = config_storage->GetValue<std::vector<Key>>(kKeys);
  }
  if (!std::any_of(keys.begin(), keys.end(), [&](const Key &key_value) { return key_value == key; })) {
    keys.push_back(key);
    config_storage->PutValue(kKeys, keys);
  }

  // Persist kernel input shape
  std::vector<std::vector<std::vector<size_t>>> shapes_list;
  if (config_storage->Exists(kShapes)) {
    shapes_list = config_storage->GetValue<std::vector<std::vector<std::vector<size_t>>>>(kShapes);
  }
  if (shapes_list.size() < keys.size()) {
    std::vector<std::vector<size_t>> shape_tmp;
    (void)std::transform(shapes->begin(), shapes->end(), std::back_inserter(shape_tmp),
                         [](const std::shared_ptr<std::vector<size_t>> &shape_ptr) { return *shape_ptr; });
    shapes_list.push_back(shape_tmp);
    config_storage->PutValue<std::vector<std::vector<std::vector<size_t>>>>(kShapes, shapes_list);
  }

  // Persist parameter name of kernel.
  std::vector<std::string> param_names;
  if (config_storage->Exists(kParamNames)) {
    param_names = config_storage->GetValue<std::vector<std::string>>(kParamNames);
  }
  const std::string &param_name = param_init_info.param_name_;
  if (param_names.size() < keys.size()) {
    param_names.push_back(param_name);
    config_storage->PutValue<std::vector<std::string>>(kParamNames, param_names);
  }
}

void ParameterServer::PersistInitParameters(const Key &key, const WeightPtr &param) {
  if (!EnableRecovery()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(server_node_);
  std::string storage_file_path = std::string(kCurrentDirOfServer) + std::to_string(server_node_->rank_id()) +
                                  std::string(kParamWithKey) + std::to_string(key);
  if (!distributed::storage::FileIOUtils::IsFileOrDirExist(storage_file_path)) {
    distributed::storage::FileIOUtils::CreateDir(storage_file_path);
  }

  auto ret = FileUtils::GetRealPath(storage_file_path.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path of persistent storage file for parameter, key: " << key;
  }

  std::string real_storage_file_path = ret.value();
  auto persistent_weight = std::dynamic_pointer_cast<PersistentWeight>(param);
  MS_EXCEPTION_IF_NULL(persistent_weight);
  std::map<std::string, std::string> config_map;
  config_map[distributed::storage::kFileStoragePath] = real_storage_file_path;
  persistent_weight->Initialize(config_map);

  (void)weights_dirty_info_.emplace(key, distributed::storage::DirtyInfo());
  persistent_weight->Persist(distributed::storage::DirtyInfo());

  MS_LOG(INFO) << "Finish persist initialized parameter, key: " << key;
}

void ParameterServer::InitEmbeddingTable(
  const Key &key, const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes,
  const ParamInitInfo &param_init_info) {
  if (EnableRecovery()) {
    while (!finish_recovery_) {
      std::this_thread::yield();
    }
  }

  std::unique_lock<std::mutex> locker(access_weight_mutex_);

  MS_EXCEPTION_IF_NULL(shapes);
  if (weights_.count(key) == 0) {
    std::shared_ptr<PServerKernel> lookup =
      std::make_shared<kernel::ps::EmbeddingLookUpPSKernelMod>(server_node_->rank_id(), pserver_num_, worker_num_);
    lookup->InitKernel(shapes);
    embedding_lookup_ops_[key] = lookup;

    PersistKernels(key, shapes, param_init_info);

    // Init embedding weight
    const std::vector<size_t> &input_shapes = lookup->input_sizes();
    size_t total_dims =
      std::accumulate(input_shapes.begin(), input_shapes.end(), IntToSize(1), std::multiplies<size_t>());

    std::shared_ptr<std::vector<int>> embedding_shape = std::make_shared<std::vector<int>>();
    (void)std::transform(input_shapes.begin(), input_shapes.end(), std::back_inserter(*embedding_shape),
                         [](size_t dim) { return static_cast<int>(dim); });

    WeightPtr embedding =
      Util::MakeWeightPtr(std::make_shared<std::vector<float>>(total_dims, 0), EnableRecovery(), embedding_shape);
    MS_EXCEPTION_IF_NULL(embedding);
    float *embedding_data = embedding->data();

    if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
      CacheEmbeddingTableParamPtr();
      if (param_init_info.param_type_ == kWeight) {
        const std::string &param_name = param_init_info.param_name_;
        auto iter = embedding_parameter_tables_.find(param_name);
        if (iter == embedding_parameter_tables_.end()) {
          MS_LOG(EXCEPTION) << "Can not find parameter info for: " << param_name;
        }
        // Cache embedding table parameter by weight key to parameter node pointer.
        (void)embedding_tables_.emplace(key, iter->second);

        InitRandomNormal(0, kStdDev, input_shapes, param_init_info.global_seed_, param_init_info.op_seed_,
                         embedding_data);
      } else if (param_init_info.param_type_ == kAccumulation) {
        InitAccumParallel(param_init_info.init_val_, total_dims, embedding_data);
      }
    } else {
      std::default_random_engine engine;
      std::normal_distribution<float> random(0, kStdDev);
      for (size_t i = 0; i < total_dims; i++) {
        embedding_data[i] = random(engine);
      }
    }

    PersistInitParameters(key, embedding);

    weights_[key] = embedding;
    MS_LOG(DEBUG) << "The key:" << key << " the embedding:" << *(embedding->MutableData());
    tokens_[key] = 0;
    is_embedding_[key] = true;

    grads_accum_counter_[key] = 0;
  }
}

bool ParameterServer::HasWeight(const Key &key) { return (weights_.count(key) > 0 && !is_embedding_.count(key)); }

void ParameterServer::Finalize() {
  running_ = false;
  apply_grads_cv_.notify_one();

  if (persist_thread_ != nullptr && persist_thread_->joinable()) {
    persist_thread_->join();
  }
}

void ParameterServer::UpdateWeights() {
  while (true) {
    MS_LOG(INFO) << "The running is:" << running_ << " the ready is:" << this->ReadyForUpdateWeights();
    std::unique_lock<std::mutex> lock(mutex_);
    apply_grads_cv_.wait(lock, [this] { return this->ReadyForUpdateWeights() || !running_; });
    if (!running_) {
      break;
    }

    for (auto iter = weights_.begin(); iter != weights_.end(); iter++) {
      Key key = iter->first;
      WeightPtr weight_ptr = iter->second;

      std::shared_ptr<PServerKernel> optimizer = nullptr;
      if (weight_key_to_optims_.count(key) > 0) {
        optimizer = optimizers_[key];
      }
      MS_EXCEPTION_IF_NULL(optimizer);

      std::shared_ptr<OptimizerInfo> optim_info = optim_infos_[key];
      if (optim_info != nullptr) {
        const std::vector<kernel::AddressPtr> &inputs = optim_info->inputs();
        const std::vector<kernel::AddressPtr> &workspaces = optim_info->workspaces();
        const std::vector<kernel::AddressPtr> &outputs = optim_info->outputs();

        std::vector<std::vector<size_t>> shapes = {};
        std::vector<size_t> indices_shape = {};
        indices_shape.emplace_back(optim_info->indice_size());
        shapes.push_back(indices_shape);

        if (original_optim_inputs_shape_.count(key) != 0) {
          std::transform((*(original_optim_inputs_shape_[key])).begin(), (*(original_optim_inputs_shape_[key])).end(),
                         std::back_inserter(shapes),
                         [](const std::shared_ptr<std::vector<size_t>> &input_shapes) -> std::vector<size_t> {
                           return *input_shapes;
                         });
        }
        optimizer->ReInit(shapes);
        optim_info->ComputeMean(shapes, worker_num_, pserver_num_, server_node_->rank_id());
        optimizer->Execute(inputs, workspaces, outputs);
        optim_info->Reset();
      }
      if (!is_embedding_[key]) {
        tokens_[key] = worker_num_;
      }
    }
    ResetGradAccumCount();
  }
}

void ParameterServer::AccumGrad(const Keys &keys, const Values &values, const Lengths &lengths) {
  std::unique_lock<std::mutex> lock(mutex_);
  const Key &key = keys[0];
  bool no_sparse_grad = values.size() == 1 && values[0] == kGradValue;
  if (!no_sparse_grad) {
    std::shared_ptr<OptimizerInfo> optim_info = optim_infos_[key];

    // Create or update the optimizer info
    if (optim_info == nullptr) {
      const std::shared_ptr<OptimizerInfoBuilder> &builder = optim_info_builders_[weight_key_to_optims_[key]];
      std::shared_ptr<kernel::ps::PServerKernel> pserver_kernel = optimizers_[key];
      if (pserver_kernel == nullptr) {
        MS_LOG(EXCEPTION) << "no optimizer found for key " << key << " optim name " << weight_key_to_optims_[key];
      }
      MS_EXCEPTION_IF_NULL(pserver_kernel);
      OptimizerInfo *optim = builder->Build(pserver_kernel, weights_[key], keys, values, lengths,
                                            optim_inputs_shape_[key], worker_num_, is_embedding_[key]);
      optim_info.reset(optim);
      optim_infos_[key] = optim_info;
    } else {
      optim_info->Update(values, lengths);
      optim_info->Accumulate(values, lengths);
    }
  }

  grads_accum_counter_[key] += 1;
  if (grads_accum_counter_[key] == worker_num_) {
    grad_accum_count_++;
  }
  if (ReadyForUpdateWeights()) {
    apply_grads_cv_.notify_one();
  }
}

WeightPtr ParameterServer::weight(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (weights_.count(key) == 0) {
    MS_LOG(EXCEPTION) << "Invalid weight key " << key;
  }
  WeightPtr weight_ptr = weights_[key];
  MS_EXCEPTION_IF_NULL(weight_ptr);
  tokens_[key] -= 1;
  return weight_ptr;
}

void ParameterServer::DoEmbeddingLookup(Key key, const LookupIds &lookup_ids, KVMessage *res) {
  if (EnableRecovery()) {
    while (!finish_recovery_) {
      std::this_thread::yield();
    }
  }

  std::unique_lock<std::mutex> lock(mutex_);
  MS_EXCEPTION_IF_NULL(res);
  if (weights_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding table key " << key;
    return;
  }
  if (embedding_lookup_ops_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding lookup op key " << key;
    return;
  }
  WeightPtr table_ptr = weights_[key];
  MS_EXCEPTION_IF_NULL(table_ptr);
  std::shared_ptr<PServerKernel> table_lookup_op = embedding_lookup_ops_[key];
  MS_EXCEPTION_IF_NULL(table_lookup_op);

  // Update shapes of lookup operator
  std::vector<std::vector<size_t>> shapes = {};
  std::vector<size_t> indices_shape = {};
  indices_shape.emplace_back(lookup_ids.size());
  shapes.push_back(indices_shape);
  table_lookup_op->ReInit(shapes);

  const std::vector<size_t> output_shapes = table_lookup_op->output_sizes();
  std::vector<kernel::AddressPtr> inputs;
  AddressPtr embedding_table = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(embedding_table);
  AddressPtr indices = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(indices);
  inputs.push_back(embedding_table);
  inputs.push_back(indices);
  embedding_table->addr = table_ptr->data();
  embedding_table->size = table_ptr->size() * sizeof(float);

  std::unique_ptr<int[]> tmp_ids = std::make_unique<int[]>(lookup_ids.size());
  MS_EXCEPTION_IF_NULL(tmp_ids);
  for (size_t i = 0; i < lookup_ids.size(); i++) {
    tmp_ids[i] = static_cast<int>(lookup_ids[i]);
  }
  indices->addr = tmp_ids.get();
  indices->size = lookup_ids.size() * sizeof(int);

  std::vector<kernel::AddressPtr> workspaces;
  std::vector<kernel::AddressPtr> outputs;
  AddressPtr output = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(output);
  std::shared_ptr<Values> addr = std::make_shared<Values>(output_shapes[0] / sizeof(float), 0);
  MS_EXCEPTION_IF_NULL(addr);

  output->addr = addr->data();
  output->size = output_shapes[0];
  outputs.push_back(output);

  table_lookup_op->Execute(inputs, workspaces, outputs);
  *res->mutable_values() = {addr->begin(), addr->end()};
  res->add_len(res->values_size());
}

void ParameterServer::UpdateEmbeddings(const Key &key, const LookupIds &lookup_ids, const Values &vals) {
  if (EnableRecovery()) {
    while (!finish_recovery_) {
      std::this_thread::yield();
    }
  }

  std::unique_lock<std::mutex> locker(access_weight_mutex_);

  if (weights_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding table key " << key;
    return;
  }
  if (embedding_lookup_ops_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding lookup op key " << key;
    return;
  }
  WeightPtr table_ptr = weights_[key];
  MS_EXCEPTION_IF_NULL(table_ptr);
  std::shared_ptr<PServerKernel> lookup_op = embedding_lookup_ops_[key];
  MS_EXCEPTION_IF_NULL(lookup_op);
  lookup_op->UpdateEmbeddings(table_ptr->data(), lookup_ids.data(), vals.data(), lookup_ids.size());

  UpdateDirtyInfo(key, lookup_ids, lookup_op->offset());
}

void ParameterServer::UpdateDirtyInfo(const Key &key, const LookupIds &lookup_ids, int64_t offset) {
  if (EnableRecovery()) {
    std::set<int> sorted_ids;
    (void)std::for_each(lookup_ids.begin(), lookup_ids.end(), [&](uint64_t id) {
      int index = SizeToInt(id) - LongToInt(offset);
      (void)sorted_ids.insert(index);
    });

    auto iter = weights_dirty_info_.find(key);
    if (iter == weights_dirty_info_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find dirty info for embedding table, key: " << key;
    }
    distributed::storage::DirtyInfo &dirty_info = iter->second;
    (void)std::for_each(sorted_ids.begin(), sorted_ids.end(), [&](int id) { dirty_info.push_back(id); });
  }
}

inline bool ParameterServer::ReadyForUpdateWeights() const {
  return grads_accum_counter_.size() > 0 && grad_accum_count_ == grads_accum_counter_.size();
}

inline bool ParameterServer::ReadyForPush(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (weights_.empty()) {
    MS_LOG(EXCEPTION) << "The weights in server is empty. Many reasons could cause this: 1.The Worker didn't send "
                         "kInitWeightsCmd command. 2.The Server failed to initialize weights.";
  }
  return grad_accum_count_ < weights_.size() && tokens_[key] == 0;
}

inline bool ParameterServer::ReadyForPull(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (tokens_.count(key) == 0 || weights_[key] == 0) {
    MS_LOG(EXCEPTION) << "Invalid weight key " << key;
  }
  MS_LOG(INFO) << "ReadyForPull: " << (tokens_[key] > 0);
  return tokens_[key] > 0;
}

inline void ParameterServer::ResetGradAccumCount() {
  grad_accum_count_ = 0;
  for (auto iter = grads_accum_counter_.begin(); iter != grads_accum_counter_.end(); iter++) {
    grads_accum_counter_[iter->first] = 0;
  }
}

const CNodePtr ParameterServer::GetCNode(const std::string &name) const {
  std::list<CNodePtr> cnodes = func_graph_->GetOrderedCnodes();
  for (CNodePtr cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::string fullname = cnode->fullname_with_scope();
    if (fullname.find(name) != std::string::npos && fullname.find("Push") != std::string::npos) {
      return cnode;
    }
  }
  return nullptr;
}

inline std::mutex &ParameterServer::mutex() { return mutex_; }

void ParameterServer::GetEmbeddingTableParamPtr() {
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph_);
  auto cnodes = func_graph_->GetOrderedCnodes();
  Key count = 0;
  for (auto cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::string cnode_name = Util::GetPrimitiveName(cnode);
    if (cnode_name == kEmbeddingLookupOpName || cnode_name == kGatherV2OpName || cnode_name == kSparseGatherV2OpName) {
      auto embedding_table = common::AnfAlgo::GetInputNode(cnode, 0);
      if (IsPrimitiveCNode(embedding_table, prim::kPrimLoad)) {
        auto embedding_cnode = embedding_table->cast<CNodePtr>();
        embedding_table = common::AnfAlgo::GetInputNode(embedding_cnode, 0);
      }
      MS_EXCEPTION_IF_NULL(embedding_table);
      if (embedding_table->isa<Parameter>()) {
        MS_LOG(INFO) << "Embedding table name is " << embedding_table->fullname_with_scope() << ", key is " << count;
        (void)embedding_tables_.emplace(count, embedding_table->cast<ParameterPtr>());
        count++;
      }
    }
  }
}

void ParameterServer::CacheEmbeddingTableParamPtr() {
  if (embedding_param_ptr_cached_) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph_);
  auto cnodes = func_graph_->GetOrderedCnodes();
  for (auto cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::string cnode_name = Util::GetPrimitiveName(cnode);
    if (cnode_name != kGatherV2OpName && cnode_name != kSparseGatherV2OpName) {
      continue;
    }

    auto embedding_table = common::AnfAlgo::GetInputNode(cnode, 0);
    if (IsPrimitiveCNode(embedding_table, prim::kPrimLoad)) {
      auto embedding_cnode = embedding_table->cast<CNodePtr>();
      embedding_table = common::AnfAlgo::GetInputNode(embedding_cnode, 0);
    }

    MS_EXCEPTION_IF_NULL(embedding_table);
    if (embedding_table->isa<Parameter>()) {
      (void)embedding_parameter_tables_.emplace(embedding_table->fullname_with_scope(),
                                                embedding_table->cast<ParameterPtr>());
    }
  }

  embedding_param_ptr_cached_ = true;
}

void ParameterServer::RecoverKernels(const std::vector<Key> &keys,
                                     const std::vector<std::vector<std::vector<size_t>>> &shapes_list,
                                     const std::vector<std::string> &param_names) {
  for (size_t i = 0; i < keys.size(); i++) {
    size_t key = keys.at(i);
    if (weights_.count(key) == 0) {
      // Recover embedding lookup kernels.
      std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> shapes_ptr =
        std::make_shared<std::vector<std::shared_ptr<std::vector<size_t>>>>();

      const auto &shapes = shapes_list[i];
      for (const auto &shape : shapes) {
        std::shared_ptr<std::vector<size_t>> shape_ptr =
          std::make_shared<std::vector<size_t>>(shape.begin(), shape.end());
        shapes_ptr->push_back(shape_ptr);
      }

      std::shared_ptr<PServerKernel> lookup =
        std::make_shared<kernel::ps::EmbeddingLookUpPSKernelMod>(server_node_->rank_id(), pserver_num_, worker_num_);
      lookup->InitKernel(shapes_ptr);
      embedding_lookup_ops_[key] = lookup;

      // Recover embedding table parameter node address in graph.
      const auto &param_name = param_names.at(i);
      auto iter = embedding_parameter_tables_.find(param_name);
      if (iter != embedding_parameter_tables_.end()) {
        // Cache embedding table parameter by weight key to parameter node pointer.
        (void)embedding_tables_.emplace(key, iter->second);
      }
    }
  }
}

void ParameterServer::RecoverParameters(const std::vector<Key> &keys) {
  for (size_t i = 0; i < keys.size(); i++) {
    size_t key = keys.at(i);
    if (weights_.count(key) == 0) {
      auto iter = embedding_lookup_ops_.find(key);
      if (iter == embedding_lookup_ops_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find embedding lookup kernel for key: " << key;
      }
      std::shared_ptr<PServerKernel> lookup = iter->second;
      MS_EXCEPTION_IF_NULL(lookup);
      const std::vector<size_t> &input_shapes = lookup->input_sizes();
      size_t total_dims =
        std::accumulate(input_shapes.begin(), input_shapes.end(), IntToSize(1), std::multiplies<size_t>());

      std::shared_ptr<std::vector<int>> embedding_shape = std::make_shared<std::vector<int>>();
      (void)std::transform(input_shapes.begin(), input_shapes.end(), std::back_inserter(*embedding_shape),
                           [](size_t dim) { return static_cast<int>(dim); });

      PersistentWeightPtr embedding =
        std::make_shared<PersistentWeight>(std::make_shared<std::vector<float>>(total_dims, 0), embedding_shape);
      MS_EXCEPTION_IF_NULL(server_node_);
      std::string storage_file_path = std::string(kCurrentDirOfServer) + std::to_string(server_node_->rank_id()) +
                                      std::string(kParamWithKey) + std::to_string(key);
      if (!distributed::storage::FileIOUtils::IsFileOrDirExist(storage_file_path)) {
        MS_LOG(EXCEPTION) << "The storage file does not exist, file path: " << storage_file_path;
      }

      auto ret = FileUtils::GetRealPath(storage_file_path.c_str());
      if (!ret.has_value()) {
        MS_LOG(EXCEPTION) << "Cannot get real path of persistent storage file for parameter, key: " << key;
      }
      std::string real_storage_file_path = ret.value();

      std::map<std::string, std::string> config_map;
      config_map[distributed::storage::kFileStoragePath] = real_storage_file_path;
      embedding->Initialize(config_map);
      embedding->Restore();
      weights_[key] = embedding;
      (void)weights_dirty_info_.emplace(key, distributed::storage::DirtyInfo());
    }
  }
}

void ParameterServer::RecoverEmbedding(const std::vector<Key> &keys,
                                       const std::vector<std::vector<std::vector<size_t>>> &shapes_list,
                                       const std::vector<std::string> &param_names) {
  CacheEmbeddingTableParamPtr();
  size_t keys_size = keys.size();
  size_t shapes_size = shapes_list.size();
  size_t params_size = param_names.size();
  if (keys_size != shapes_size || keys_size != params_size) {
    MS_LOG(EXCEPTION) << "Bad input parameter number, keys_size: " << keys_size << ", shapes_size: " << shapes_size
                      << ", params_size: " << params_size;
  }

  RecoverKernels(keys, shapes_list, param_names);
  RecoverParameters(keys);
}

void ParameterServer::set_persistent_state(core::PersistentState persistent_state) const {
  MS_EXCEPTION_IF_NULL(server_node_);
  server_node_->set_persistent_state(persistent_state);
}

bool ParameterServer::EnableRecovery() const {
  MS_EXCEPTION_IF_NULL(server_node_);
  return server_node_->EnableRecovery();
}

void ParameterServer::PersistParameters() {
  if (!EnableRecovery() || !finish_recovery_) {
    return;
  }

  if (persist_thread_ != nullptr && persist_thread_->joinable()) {
    persist_thread_->join();
  }

  auto do_persist_task = [this]() {
    std::unique_lock<std::mutex> locker(access_weight_mutex_);

    set_persistent_state(core::PersistentState::PERSISTING);

    for (const auto &weight_key_pair : weights_) {
      const WeightPtr &weight = weight_key_pair.second;
      auto persistent_weight = std::dynamic_pointer_cast<PersistentWeight>(weight);
      MS_EXCEPTION_IF_NULL(persistent_weight);

      Key key = weight_key_pair.first;
      auto iter = weights_dirty_info_.find(key);
      if (iter == weights_dirty_info_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find dirty info for weight, key: " << key;
      }

      distributed::storage::DirtyInfo &dirty_info = iter->second;
      persistent_weight->Persist(dirty_info);

      dirty_info.clear();
    }

    set_persistent_state(core::PersistentState::FINISH_PERSIST);
    MS_LOG(INFO) << "Finish persist weights in parameter server";
  };

  persist_thread_ = std::make_unique<std::thread>(do_persist_task);
}

void ParameterServer::SyncEmbeddingTables() {
  for (auto embedding_table : embedding_tables_) {
    Key key = embedding_table.first;
    if (embedding_lookup_ops_.count(key) == 0) {
      MS_LOG(WARNING) << "Can't find look up PS kernel for key " << key;
      continue;
    }
    auto lookup = embedding_lookup_ops_[key];
    const std::vector<size_t> &input_shapes = lookup->input_sizes();
    std::vector<int64_t> new_tensor_shape(input_shapes.begin(), input_shapes.end());

    tensor::TensorPtr new_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, new_tensor_shape);
    MS_EXCEPTION_IF_NULL(new_tensor);
    float *new_tensor_data_ptr = reinterpret_cast<float *>(new_tensor->data_c());
    size_t new_tensor_size = static_cast<size_t>(new_tensor->data().nbytes());
    size_t embedding_table_size = weights_[key]->size() * sizeof(float);
    if (new_tensor_size != embedding_table_size) {
      MS_LOG(EXCEPTION) << "Shape of embedding table can't match. New tensor size:" << new_tensor_size
                        << ", embedding_table size:" << embedding_table_size;
    }
    MS_EXCEPTION_IF_NULL(new_tensor_data_ptr);
    MS_EXCEPTION_IF_NULL(weights_[key]->data());

    CopyTensorData(new_tensor_data_ptr, new_tensor_size, weights_[key]->data());

    auto paramter_tensor_ptr = embedding_table.second->default_param();
    MS_EXCEPTION_IF_NULL(paramter_tensor_ptr);
    paramter_tensor_ptr->cast<tensor::TensorPtr>()->AssignValue(*new_tensor);
  }
}

void ParameterServer::ServerHandler::Init() {
  handlers_[kInitWeightsCmd] = &ServerHandler::HandleInitWeights;
  handlers_[kInitWeightToOptimIdCmd] = &ServerHandler::HandleInitWeightToOptimId;
  handlers_[kInitOptimInputsShapeCmd] = &ServerHandler::HandleInitInputsShape;
  handlers_[kInitEmbeddingsCmd] = &ServerHandler::HandleInitEmbeddings;
  handlers_[kCheckReadyForPushCmd] = &ServerHandler::HandleCheckReadyForPush;
  handlers_[kCheckReadyForPullCmd] = &ServerHandler::HandleCheckReadyForPull;
  handlers_[kEmbeddingLookupCmd] = &ServerHandler::HandleEmbeddingLookup;
  handlers_[kUpdateEmbeddingsCmd] = &ServerHandler::HandleUpdateEmbeddings;
  handlers_[kFinalizeCmd] = &ServerHandler::HandleFinalize;
  handlers_[kPushCmd] = &ServerHandler::HandlePushReq;
  handlers_[kPullCmd] = &ServerHandler::HandlePullReq;
  commands_[kInitWeightsCmd] = "kInitWeightsCmd";
  commands_[kInitWeightToOptimIdCmd] = "kInitWeightToOptimIdCmd";
  commands_[kInitOptimInputsShapeCmd] = "kInitOptimInputsShapeCmd";
  commands_[kInitEmbeddingsCmd] = "kInitEmbeddingsCmd";
  commands_[kCheckReadyForPushCmd] = "kCheckReadyForPushCmd";
  commands_[kCheckReadyForPullCmd] = "kCheckReadyForPullCmd";
  commands_[kEmbeddingLookupCmd] = "kEmbeddingLookupCmd";
  commands_[kUpdateEmbeddingsCmd] = "kUpdateEmbeddingsCmd";
  commands_[kFinalizeCmd] = "kFinalizeCmd";
  commands_[kPushCmd] = "kPushCmd";
  commands_[kPullCmd] = "kPullCmd";
}

void ParameterServer::ServerHandler::operator()(const std::shared_ptr<core::TcpConnection> &conn,
                                                const std::shared_ptr<core::MessageMeta> &meta, const void *data,
                                                size_t size) {
  MS_EXCEPTION_IF_NULL(data);
  auto output = std::make_shared<std::vector<unsigned char>>();
  if (commands_.count(meta->user_cmd()) == 0) {
    MS_LOG(EXCEPTION) << "The command:" << meta->user_cmd() << " is not supported!";
  }
  MS_LOG(INFO) << "The command is:" << commands_[meta->user_cmd()];

  auto &handler_ptr = handlers_[meta->user_cmd()];
  (this->*handler_ptr)(data, size, output);
  MS_LOG(DEBUG) << "The output size is:" << output->size();

  if (output->size() > 0) {
    ps_->server_node_->Response(conn, meta, output->data(), output->size());
  } else {
    // If the size of the output is 0, then constructed an empty string, Because the Response function is a synchronous,
    // the res variable  will be automatically recycled after calling the Response function
    std::string res;
    ps_->server_node_->Response(conn, meta, res.data(), res.length());
  }
  MS_LOG(DEBUG) << "The request id is:" << meta->request_id() << " the current time is:"
                << std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now())
                     .time_since_epoch()
                     .count();
}

void ParameterServer::ServerHandler::HandlePushReq(const void *data, size_t size, const VectorPtr &res) {
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  Keys keys = {input.keys().begin(), input.keys().end()};
  Values values = {input.values().begin(), input.values().end()};
  Lengths lens = {input.len().begin(), input.len().end()};
  MS_LOG(DEBUG) << "The keys:" << keys << " the values:" << values << " the len:" << lens;
  ps_->AccumGrad(keys, values, lens);
}

void ParameterServer::ServerHandler::HandlePullReq(const void *data, size_t size, const VectorPtr &res) {
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  KVMessage res_data;
  *res_data.mutable_keys() = input.keys();
  Key key = input.keys()[0];
  auto weight = ps_->weight(key);
  auto weight_data = weight->MutableData();
  MS_EXCEPTION_IF_NULL(weight_data);
  *res_data.mutable_values() = {weight_data->begin(), weight_data->end()};
  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleInitWeights(const void *data, size_t size, const VectorPtr &res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  int key_num = input.keys_size();
  const float *data_ptr = input.values().data();
  size_t pos = 0;
  for (int i = 0; i < key_num; i++) {
    Key key = input.keys()[i];
    size_t data_len = input.len_size() != key_num ? input.values_size() / key_num : input.len()[i];

    if (!ps_->HasWeight(key)) {
      WeightPtr weight_ptr = Util::MakeWeightPtr(
        std::make_shared<std::vector<float>>(data_ptr + pos, data_ptr + (pos + data_len)), ps_->EnableRecovery());
      MS_EXCEPTION_IF_NULL(weight_ptr);
      ps_->InitWeight(key, weight_ptr);

      GradPtr grad_ptr = std::make_shared<std::vector<float>>(data_len, 0);
      MS_EXCEPTION_IF_NULL(grad_ptr);
      ps_->InitGrad(key, grad_ptr);
    }
    pos += data_len;
  }
}

void ParameterServer::ServerHandler::HandleInitWeightToOptimId(const void *data, size_t size, const VectorPtr &res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  int key_num = input.keys_size();
  for (int i = 0; i < key_num; i++) {
    Key key = input.keys()[i];
    float val = input.values()[i];
    if (init_weight_to_optim_[key]) {
      continue;
    } else {
      init_weight_to_optim_[key] = true;
    }
    ps_->InitWeightKeyToOptims(key, static_cast<int64_t>(val));
  }
}

void ParameterServer::ServerHandler::HandleInitInputsShape(const void *data, size_t size, const VectorPtr &res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  const Key &key = input.keys()[0];
  if (init_optim_info_[key]) {
    return;
  } else {
    init_optim_info_[key] = true;
  }
  Keys keys = {input.keys().begin(), input.keys().end()};
  Values values = {input.values().begin(), input.values().end()};
  Lengths lens = {input.len().begin(), input.len().end()};
  ps_->InitOptimInputsShape(keys, values, lens);
}

void ParameterServer::ServerHandler::HandleInitEmbeddings(const void *data, size_t size, const VectorPtr &) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(data);
  EmbeddingTableMeta embedding_table_meta;
  CHECK_RETURN_TYPE(embedding_table_meta.ParseFromArray(data, SizeToInt(size)));
  const Key &key = embedding_table_meta.key();
  MS_LOG(INFO) << "Initializing embedding table for key:" << key;
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> shapes =
    std::make_shared<std::vector<std::shared_ptr<std::vector<size_t>>>>();
  MS_EXCEPTION_IF_NULL(shapes);
  std::shared_ptr<std::vector<size_t>> input_shape = std::make_shared<std::vector<size_t>>(
    embedding_table_meta.input_shape().begin(), embedding_table_meta.input_shape().end());
  MS_EXCEPTION_IF_NULL(input_shape);
  std::shared_ptr<std::vector<size_t>> indices_shape = std::make_shared<std::vector<size_t>>(
    embedding_table_meta.indices_shape().begin(), embedding_table_meta.indices_shape().end());
  MS_EXCEPTION_IF_NULL(indices_shape);
  std::shared_ptr<std::vector<size_t>> output_shape = std::make_shared<std::vector<size_t>>(
    embedding_table_meta.output_shape().begin(), embedding_table_meta.output_shape().end());
  MS_EXCEPTION_IF_NULL(output_shape);
  shapes->push_back(input_shape);
  shapes->push_back(indices_shape);
  shapes->push_back(output_shape);

  const ParamInitInfoMessage &info = embedding_table_meta.info();
  ParamInitInfo param_init_info;
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    param_init_info.param_name_ = info.param_name();
    param_init_info.param_type_ = static_cast<ParamType>(info.param_type());
    if (param_init_info.param_type_ == kWeight) {
      param_init_info.global_seed_ = info.global_seed();
      param_init_info.op_seed_ = info.op_seed();
    } else if (param_init_info.param_type_ == kAccumulation) {
      param_init_info.init_val_ = info.init_val();
    }
  }
  ps_->InitEmbeddingTable(key, shapes, param_init_info);
}

void ParameterServer::ServerHandler::HandleCheckReadyForPush(const void *data, size_t size, const VectorPtr &res) {
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  const Key &key = input.keys()[0];
  bool ready = ps_->ReadyForPush(key);
  MS_LOG(INFO) << "The ready is:" << ready;
  KVMessage res_data;
  res_data.add_keys(key);
  res_data.add_values(ready);
  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleCheckReadyForPull(const void *data, size_t size, const VectorPtr &res) {
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  const Key &key = input.keys()[0];
  bool ready = ps_->ReadyForPull(key);
  KVMessage res_data;
  res_data.add_keys(key);
  res_data.add_values(ready);
  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleEmbeddingLookup(const void *data, size_t size, const VectorPtr &res) {
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  EmbeddingTableLookup input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  const Key &key = input.key();

  KVMessage res_data;
  std::vector<Key> keys = {input.keys().begin(), input.keys().end()};
  *res_data.mutable_keys() = {input.keys().begin(), input.keys().end()};

  ps_->DoEmbeddingLookup(key, keys, &res_data);

  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleUpdateEmbeddings(const void *data, size_t size, const VectorPtr &res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  CHECK_RETURN_TYPE(input.ParseFromArray(data, SizeToInt(size)));
  const Key &key = input.keys()[0];
  const LookupIds &lookup_ids = {input.keys().begin() + 1, input.keys().end()};
  const Values &update_vals = {input.values().begin(), input.values().end()};
  ps_->UpdateEmbeddings(key, lookup_ids, update_vals);
}

void ParameterServer::ServerHandler::HandleFinalize(const void *, size_t, const VectorPtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  ps_->Finalize();
}

void ParameterServer::RecoverHandler::Init() {
  handlers_[kRecoverEmbedding] = &RecoverHandler::RecoverEmbedding;

  MS_EXCEPTION_IF_NULL(ps_);
  MS_EXCEPTION_IF_NULL(ps_->server_node_);
  std::string persistent_storage_file_path =
    std::string(kCurrentDirOfServer) + std::to_string(ps_->server_node_->rank_id()) + "_persistent_storage.json";
  storage_ = std::make_unique<core::FileConfiguration>(persistent_storage_file_path);
  (void)storage_->Initialize();
}

void ParameterServer::RecoverHandler::Recover() {
  MS_EXCEPTION_IF_NULL(storage_);
  if (!storage_->Exists(kRecoverFunc)) {
    return;
  }

  std::vector<std::string> func_names = storage_->GetValue<std::vector<std::string>>(kRecoverFunc);
  for (const auto &func_name : func_names) {
    if (func_name.empty()) {
      MS_LOG(EXCEPTION) << "The recover function name is empty";
    }

    auto iter = handlers_.find(func_name);
    if (iter == handlers_.end()) {
      MS_LOG(EXCEPTION) << "Can not find func: [" << func_name << "]";
    }
    auto &fun_ptr = iter->second;
    MS_EXCEPTION_IF_NULL(fun_ptr);
    (this->*fun_ptr)();
  }
}

void ParameterServer::RecoverHandler::RecoverEmbedding() {
  MS_EXCEPTION_IF_NULL(storage_);
  std::vector<Key> keys = storage_->GetValue<std::vector<Key>>(kKeys);
  std::vector<std::vector<std::vector<size_t>>> shapes_list =
    storage_->GetValue<std::vector<std::vector<std::vector<size_t>>>>(kShapes);
  std::vector<std::string> param_names = storage_->GetValue<std::vector<std::string>>(kParamNames);

  MS_EXCEPTION_IF_NULL(ps_);
  ps_->RecoverEmbedding(keys, shapes_list, param_names);
}
}  // namespace ps
}  // namespace mindspore
