/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/source/en_wik9_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/en_wik9_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for EnWik9Node
EnWik9Node::EnWik9Node(const std::string &dataset_dir, int32_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                       int32_t shard_id, const std::shared_ptr<DatasetCache> &cache)
    : NonMappableSourceNode(cache),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id),
      dataset_dir_(dataset_dir) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass. User discretion
  // is advised. Auto_num_worker_pass is currently an experimental feature which can still work if the num_shards_ isn't
  // 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to return num_shards. Once
  // PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
  DirToPath(dataset_dir_);
}

std::shared_ptr<DatasetNode> EnWik9Node::Copy() {
  auto node = std::make_shared<EnWik9Node>(dataset_dir_, num_samples_, shuffle_, num_shards_, shard_id_, cache_);
  node->SetNumWorkers(num_workers_);
  node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void EnWik9Node::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

Status EnWik9Node::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("EnWik9Dataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateEnum("EnWik9Dataset", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));
  RETURN_IF_NOT_OK(ValidateScalar("EnWik9Dataset", "num_samples", num_samples_, {0}, false));
  RETURN_IF_NOT_OK(ValidateDatasetShardParams("EnWik9Dataset", num_shards_, shard_id_));

  return Status::OK();
}

void EnWik9Node::DirToPath(const std::string &dataset_dir) {
  Path train_prefix("enwik9");
  Path dir(dataset_dir);
  Path temp_path = dir / train_prefix;
  src_target_file_list_.push_back(temp_path.ToString());
}

// Function to build EnWik9Node
Status EnWik9Node::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("text", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));

  // Create and initialize EnWik9Op
  std::shared_ptr<EnWik9Op> en_wik9_op =
    std::make_shared<EnWik9Op>(num_workers_, num_samples_, worker_connector_size_, std::move(schema),
                               src_target_file_list_, connector_que_size_, shuffle_files, num_shards_, shard_id_);
  RETURN_IF_NOT_OK(en_wik9_op->Init());

  // If a global shuffle is used for EnWik9, it will inject a shuffle op over the EnWik9.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be built.
  // This is achieved in the cache transform pass where we call MakeSimpleProducer to reset EnWik9's shuffle
  // option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_IF_NOT_OK(EnWik9Op::CountAllFileRows(src_target_file_list_, &num_rows));

    // Add the shuffle op after this op
    RETURN_IF_NOT_OK(
      AddShuffleOp(src_target_file_list_.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  en_wik9_op->SetTotalRepeats(GetTotalRepeats());
  en_wik9_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  // Add EnWik9Op
  node_ops->push_back(en_wik9_op);
  return Status::OK();
}

// Get the shard id of node
Status EnWik9Node::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;
  return Status::OK();
}

// Get Dataset size
Status EnWik9Node::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size = num_samples_;
  RETURN_IF_NOT_OK(EnWik9Op::CountAllFileRows(src_target_file_list_, &num_rows));
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status EnWik9Node::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["num_samples"] = num_samples_;
  args["shuffle"] = shuffle_;
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

// Note: The following two functions are common among NonMappableSourceNode and should be promoted to its parent class.
// EnWik9 by itself is a non-mappable dataset that does not support sampling.
// However, if a cache operator is injected at some other place higher in the tree, that cache can
// inherit this sampler from the leaf, providing sampling support from the caching layer.
// That is why we setup the sampler for a leaf node that does not use sampling.
Status EnWik9Node::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this EnWik9 node, then the cache will be executing
// a sampler for fetching the data. As such, any options in the EnWik9 node need to be reset to its defaults so
// that this EnWik9 node will produce the full set of data into the cache.
Status EnWik9Node::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
