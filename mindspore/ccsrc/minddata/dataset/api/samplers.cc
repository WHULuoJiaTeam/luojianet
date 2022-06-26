/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/dataset/samplers.h"

#include <utility>

#include "minddata/dataset/engine/ir/datasetops/source/samplers/distributed_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/pk_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/random_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/sequential_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/subset_random_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/subset_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/weighted_random_sampler_ir.h"

namespace mindspore {
namespace dataset {
Status Sampler::BuildChildren(std::shared_ptr<SamplerObj> *const sampler) const {
  for (const auto &child : children_) {
    std::shared_ptr<SamplerObj> sampler_obj = child->Parse();
    RETURN_IF_NOT_OK((*sampler)->AddChildSampler(sampler_obj));
  }
  return Status::OK();
}

// DistributedSampler
DistributedSampler::DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle, int64_t num_samples,
                                       uint32_t seed, int64_t offset, bool even_dist)
    : num_shards_(num_shards),
      shard_id_(shard_id),
      shuffle_(shuffle),
      num_samples_(num_samples),
      seed_(seed),
      offset_(offset),
      even_dist_(even_dist) {}

std::shared_ptr<SamplerObj> DistributedSampler::Parse() const {
  std::shared_ptr<SamplerObj> output =
    std::make_shared<DistributedSamplerObj>(num_shards_, shard_id_, shuffle_, num_samples_, seed_, offset_, even_dist_);
  Status s = BuildChildren(&output);
  if (s.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR] Error in Parse. Message: " << s;
  }
  return output;
}

// PKSampler
PKSampler::PKSampler(int64_t num_val, bool shuffle, int64_t num_samples)
    : num_val_(num_val), shuffle_(shuffle), num_samples_(num_samples) {}

std::shared_ptr<SamplerObj> PKSampler::Parse() const {
  std::shared_ptr<SamplerObj> output = std::make_shared<PKSamplerObj>(num_val_, shuffle_, num_samples_);
  Status s = BuildChildren(&output);
  if (s.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR] Error in Parse. Message: " << s;
  }
  return output;
}

// RandomSampler
RandomSampler::RandomSampler(bool replacement, int64_t num_samples)
    : replacement_(replacement), num_samples_(num_samples) {}

std::shared_ptr<SamplerObj> RandomSampler::Parse() const {
  std::shared_ptr<SamplerObj> output = std::make_shared<RandomSamplerObj>(replacement_, num_samples_);
  Status s = BuildChildren(&output);
  if (s.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR] Error in Parse. Message: " << s;
  }
  return output;
}

// SequentialSampler
SequentialSampler::SequentialSampler(int64_t start_index, int64_t num_samples)
    : start_index_(start_index), num_samples_(num_samples) {}

std::shared_ptr<SamplerObj> SequentialSampler::Parse() const {
  std::shared_ptr<SamplerObj> output = std::make_shared<SequentialSamplerObj>(start_index_, num_samples_);
  Status s = BuildChildren(&output);
  if (s.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR] Error in Parse. Message: " << s;
  }
  return output;
}

// SubsetSampler
SubsetSampler::SubsetSampler(const std::vector<int64_t> &indices, int64_t num_samples)
    : indices_(indices), num_samples_(num_samples) {}

std::shared_ptr<SamplerObj> SubsetSampler::Parse() const {
  std::shared_ptr<SamplerObj> output = std::make_shared<SubsetSamplerObj>(indices_, num_samples_);
  Status s = BuildChildren(&output);
  if (s.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR] Error in Parse. Message: " << s;
  }
  return output;
}

// SubsetRandomSampler
SubsetRandomSampler::SubsetRandomSampler(const std::vector<int64_t> &indices, int64_t num_samples)
    : SubsetSampler(indices, num_samples) {}

std::shared_ptr<SamplerObj> SubsetRandomSampler::Parse() const {
  std::shared_ptr<SamplerObj> output = std::make_shared<SubsetRandomSamplerObj>(indices_, num_samples_);
  Status s = BuildChildren(&output);
  if (s.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR] Error in Parse. Message: " << s;
  }
  return output;
}

// WeightedRandomSampler
WeightedRandomSampler::WeightedRandomSampler(const std::vector<double> &weights, int64_t num_samples, bool replacement)
    : weights_(weights), num_samples_(num_samples), replacement_(replacement) {}

std::shared_ptr<SamplerObj> WeightedRandomSampler::Parse() const {
  std::shared_ptr<SamplerObj> output = std::make_shared<WeightedRandomSamplerObj>(weights_, num_samples_, replacement_);
  Status s = BuildChildren(&output);
  if (s.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR] Error in Parse. Message: " << s;
  }
  return output;
}
}  // namespace dataset
}  // namespace mindspore
