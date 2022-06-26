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

#include "minddata/dataset/engine/ir/datasetops/source/photo_tour_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/photo_tour_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
PhotoTourNode::PhotoTourNode(const std::string &dataset_dir, const std::string &name, const std::string &usage,
                             std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), name_(name), usage_(usage), sampler_(sampler) {}

std::shared_ptr<DatasetNode> PhotoTourNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<PhotoTourNode>(dataset_dir_, name_, usage_, sampler, cache_);
  node->SetNumWorkers(num_workers_);
  node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void PhotoTourNode::Print(std::ostream &out) const {
  out << (Name() + "(name: " + name_ + ", usage: " + usage_);
  if (sampler_ != nullptr) {
    out << ", sampler";
  }
  if (cache_ != nullptr) {
    out << ", cache";
  }
  out << ")";
}

Status PhotoTourNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("PhotoTourNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("PhotoTourNode", sampler_));
  RETURN_IF_NOT_OK(ValidateStringValue("PhotoTourNode", usage_, {"train", "test"}));
  RETURN_IF_NOT_OK(
    ValidateStringValue("PhotoTourNode", name_,
                        {"notredame", "yosemite", "liberty", "notredame_harris", "yosemite_harris", "liberty_harris"}));

  return Status::OK();
}

Status PhotoTourNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  if (usage_ == "train") {
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  } else {
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image1", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image2", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
    TensorShape scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("matches", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  }

  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<PhotoTourOp>(dataset_dir_, name_, usage_, num_workers_, connector_que_size_,
                                          std::move(schema), std::move(sampler_rt));
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Get the shard id of node.
Status PhotoTourNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size.
Status PhotoTourNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                     int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(PhotoTourOp::CountTotalRows(dataset_dir_, name_, usage_, &num_rows));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status PhotoTourNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["name"] = name_;
  args["usage"] = usage_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
