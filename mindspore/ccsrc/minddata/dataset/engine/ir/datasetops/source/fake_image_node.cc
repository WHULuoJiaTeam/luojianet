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

#include "minddata/dataset/engine/ir/datasetops/source/fake_image_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/fake_image_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
FakeImageNode::FakeImageNode(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes,
                             int32_t base_seed, std::shared_ptr<SamplerObj> sampler,
                             std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      num_images_(num_images),
      image_size_(image_size),
      num_classes_(num_classes),
      base_seed_(base_seed),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> FakeImageNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<FakeImageNode>(num_images_, image_size_, num_classes_, base_seed_, sampler, cache_);
  node->SetNumWorkers(num_workers_);
  node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void FakeImageNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status FakeImageNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());

  RETURN_IF_NOT_OK(ValidateDatasetSampler("FakeImageDataset", sampler_));
  RETURN_IF_NOT_OK(ValidateScalar("FakeImageDataset", "num_images", num_images_, {0}, true));

  if (image_size_.size() != 3) {
    std::string err_msg = "FakeImageDataset: 'image_size' expecting size 3, but got image_size.size(): " +
                          std::to_string(image_size_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  for (auto i = 0; i < 3; i++) {
    RETURN_IF_NOT_OK(
      ValidateScalar("FakeImageDataset", "image_size[" + std::to_string(i) + "]", image_size_[i], {0}, true));
  }

  RETURN_IF_NOT_OK(ValidateScalar("FakeImageDataset", "num_classes", num_classes_, {0}, true));
  return Status::OK();
}

Status FakeImageNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<FakeImageOp>(num_images_, image_size_, num_classes_, base_seed_, num_workers_,
                                          connector_que_size_, std::move(schema), std::move(sampler_rt));
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Get the shard id of node
Status FakeImageNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status FakeImageNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                     int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  int64_t num_rows, sample_size;
  num_rows = num_images_;
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

Status FakeImageNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["num_images"] = num_images_;
  args["image_size"] = image_size_;
  args["num_classes"] = num_classes_;
  args["base_seed"] = base_seed_;
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
