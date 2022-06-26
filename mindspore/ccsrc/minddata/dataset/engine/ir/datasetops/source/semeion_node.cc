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
#include "minddata/dataset/engine/ir/datasetops/source/semeion_node.h"

#include <fstream>
#include <utility>

#include "minddata/dataset/engine/datasetops/source/semeion_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for SemeionNode.
SemeionNode::SemeionNode(const std::string &dataset_dir, const std::shared_ptr<SamplerObj> &sampler,
                         const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), sampler_(sampler) {}

std::shared_ptr<DatasetNode> SemeionNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<SemeionNode>(dataset_dir_, sampler, cache_);
  node->SetNumWorkers(num_workers_);
  node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void SemeionNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status SemeionNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("SemeionNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("SemeionNode", sampler_));
  return Status::OK();
}

Status SemeionNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape label_scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &label_scalar)));

  // Argument that is not exposed to user in the API.
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto semeion_op = std::make_shared<SemeionOp>(dataset_dir_, num_workers_, std::move(schema), std::move(sampler_rt),
                                                connector_que_size_);
  semeion_op->SetTotalRepeats(GetTotalRepeats());
  semeion_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(semeion_op);
  return Status::OK();
}

Status SemeionNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

Status SemeionNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                   int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size = -1;
  int64_t num_rows = 0;

  RETURN_IF_NOT_OK(SemeionOp::CountTotalRows(dataset_dir_, &num_rows));

  // give sampler the total number of files and check if num_samples is smaller.
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  // We cache dataset size so as to not duplicated run.
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status SemeionNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
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
