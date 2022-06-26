/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"

#include <algorithm>
#include <utility>

#include "minddata/dataset/engine/datasetops/source/csv_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for CSVNode
CSVNode::CSVNode(const std::vector<std::string> &csv_files, char field_delim,
                 const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                 const std::vector<std::string> &column_names, int64_t num_samples, ShuffleMode shuffle,
                 int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_files_(csv_files),
      field_delim_(field_delim),
      column_defaults_(column_defaults),
      column_names_(column_names),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass. User discretion
  // is advised. Auto_num_worker_pass is currently an experimental feature which can still work if the num_shards_ isn't
  // 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to return num_shards. Once
  // PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
}

std::shared_ptr<DatasetNode> CSVNode::Copy() {
  auto node = std::make_shared<CSVNode>(dataset_files_, field_delim_, column_defaults_, column_names_, num_samples_,
                                        shuffle_, num_shards_, shard_id_, cache_);
  node->SetNumWorkers(num_workers_);
  node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void CSVNode::Print(std::ostream &out) const {
  out << (Name() + "(cache:" + ((cache_ != nullptr) ? "true" : "false") + ",..." +
          ",num_shards:" + std::to_string(num_shards_) + ",shard_id:" + std::to_string(shard_id_) + ")");
}

Status CSVNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("CSVNode", dataset_files_));

  if (field_delim_ == '"' || field_delim_ == '\r' || field_delim_ == '\n') {
    std::string err_msg = "CSVNode: The field delimiter should not be \", \\r, \\n";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateEnum("CSVDataset", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));
  RETURN_IF_NOT_OK(ValidateScalar("CSVDataset", "num_samples", num_samples_, {0}, false));
  RETURN_IF_NOT_OK(ValidateDatasetShardParams("CSVDataset", num_shards_, shard_id_));

  if (find(column_defaults_.begin(), column_defaults_.end(), nullptr) != column_defaults_.end()) {
    std::string err_msg = "CSVDataset: column_default should not be null.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("CSVDataset", "column_names", column_names_));
  }

  return Status::OK();
}

// Function to build CSVNode
Status CSVNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files = dataset_files_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default_list;
  for (auto v : column_defaults_) {
    if (v->type == CsvType::INT) {
      column_default_list.push_back(
        std::make_shared<CsvOp::Record<int>>(CsvOp::INT, std::dynamic_pointer_cast<CsvRecord<int>>(v)->value));
    } else if (v->type == CsvType::FLOAT) {
      column_default_list.push_back(
        std::make_shared<CsvOp::Record<float>>(CsvOp::FLOAT, std::dynamic_pointer_cast<CsvRecord<float>>(v)->value));
    } else if (v->type == CsvType::STRING) {
      column_default_list.push_back(std::make_shared<CsvOp::Record<std::string>>(
        CsvOp::STRING, (std::dynamic_pointer_cast<CsvRecord<std::string>>(v))->value));
    }
  }

  std::shared_ptr<CsvOp> csv_op = std::make_shared<CsvOp>(
    sorted_dataset_files, field_delim_, column_default_list, column_names_, num_workers_, num_samples_,
    worker_connector_size_, connector_que_size_, shuffle_files, num_shards_, shard_id_);

  RETURN_IF_NOT_OK(csv_op->Init());

  // If a global shuffle is used for CSV, it will inject a shuffle op over the CSV.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be built.
  // This is achieved in the cache transform pass where we call MakeSimpleProducer to reset CSV's shuffle
  // option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_IF_NOT_OK(CsvOp::CountAllFileRows(sorted_dataset_files, column_names_.empty(), &num_rows));

    // Add the shuffle op after this op
    RETURN_IF_NOT_OK(
      AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  csv_op->SetTotalRepeats(GetTotalRepeats());
  csv_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(csv_op);

  return Status::OK();
}

// Get the shard id of node
Status CSVNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;

  return Status::OK();
}

// Get Dataset size
Status CSVNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                               int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(CsvOp::CountAllFileRows(dataset_files_, column_names_.empty(), &num_rows));
  sample_size = num_samples_;
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status CSVNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_files"] = dataset_files_;
  args["field_delim"] = std::string(1, field_delim_);
  args["column_names"] = column_names_;
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

Status CSVNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_files", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "field_delim", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "column_names", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_samples", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "shuffle", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_shards", kCSVNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "shard_id", kCSVNode));
  std::vector<std::string> dataset_files = json_obj["dataset_files"];
  std::string field_delim = json_obj["field_delim"];
  std::vector<std::shared_ptr<CsvBase>> column_defaults = {};
  std::vector<std::string> column_names = json_obj["column_names"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<CSVNode>(dataset_files, field_delim.c_str()[0], column_defaults, column_names, num_samples,
                                  shuffle, num_shards, shard_id, cache);
  (void)(*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  (void)(*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]);
  return Status::OK();
}

// Note: The following two functions are common among NonMappableSourceNode and should be promoted to its parent class.
// CSV by itself is a non-mappable dataset that does not support sampling.
// However, if a cache operator is injected at some other place higher in the tree, that cache can
// inherit this sampler from the leaf, providing sampling support from the caching layer.
// That is why we setup the sampler for a leaf node that does not use sampling.
Status CSVNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this CSV node, then the cache will be executing
// a sampler for fetching the data.  As such, any options in the CSV node need to be reset to its defaults so
// that this CSV node will produce the full set of data into the cache.
Status CSVNode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
