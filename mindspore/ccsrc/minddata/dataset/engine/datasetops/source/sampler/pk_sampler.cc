/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include <algorithm>
#include <memory>
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
PKSamplerRT::PKSamplerRT(int64_t num_val, bool shuffle, int64_t num_samples, int64_t samples_per_tensor)
    : SamplerRT(num_samples, samples_per_tensor),
      shuffle_(shuffle),
      seed_(GetSeed()),
      next_id_(0),
      samples_per_class_(num_val) {}

Status PKSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  labels_.reserve(label_to_ids_.size());
  for (const auto &pair : label_to_ids_) {
    if (!pair.second.empty()) {
      labels_.push_back(pair.first);
    }
  }
  rnd_.seed(seed_++);

  // The special handshake gives the list of classes and id's, but it did not set the num_rows_ to
  // capture the total number of possible sample ids.
  // Compute that here for this case to find the total number of samples that are available to return.
  // (in this case, samples per class * total classes).
  if (samples_per_class_ > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(labels_.size())) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Overflow in counting num_rows");
  }
  num_rows_ = samples_per_class_ * static_cast<int64_t>(labels_.size());

  // The user may have chosen to sample less than the total amount.
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }

  samples_per_tensor_ = (samples_per_tensor_ > num_samples_) ? num_samples_ : samples_per_tensor_;
  if (shuffle_ == true) {
    std::shuffle(labels_.begin(), labels_.end(), rnd_);
  } else {
    std::sort(labels_.begin(), labels_.end());
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_samples_ > 0, "Invalid parameter, num_class or num samples per class must be greater than 0, but got " +
                        std::to_string(num_samples_));
  is_initialized = true;
  return Status::OK();
}

Status PKSamplerRT::GetNextSample(TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  if (next_id_ > num_samples_ || num_samples_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "[Internal ERROR] Sampler index must be less than or equal to num_samples(total rows in dataset), but got: " +
      std::to_string(next_id_) + ", num_samplers:" + std::to_string(num_samples_));
  } else if (next_id_ == num_samples_) {
    (*out) = TensorRow(TensorRow::kFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    std::shared_ptr<Tensor> sample_ids;
    int64_t last_id = (samples_per_tensor_ + next_id_ > num_samples_) ? num_samples_ : samples_per_tensor_ + next_id_;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sample_ids, last_id - next_id_));
    auto id_ptr = sample_ids->begin<int64_t>();
    CHECK_FAIL_RETURN_UNEXPECTED(samples_per_class_ != 0, "Invalid Parameter, num samples per class can't be zero.");
    while (next_id_ < last_id && id_ptr != sample_ids->end<int64_t>()) {
      int64_t cls_id = next_id_++ / samples_per_class_;
      const std::vector<int64_t> &samples = label_to_ids_[labels_[cls_id]];
      int64_t rnd_ind = std::uniform_int_distribution<int64_t>(0, samples.size() - 1)(rnd_);
      int64_t sampled_id = samples[rnd_ind];

      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *id_ptr = sampled_id;
      ++id_ptr;
    }

    (*out) = {sample_ids};
  }
  return Status::OK();
}

Status PKSamplerRT::ResetSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(next_id_ == num_samples_, "[Internal ERROR] Reset() Sampler called early or late.");
  next_id_ = 0;
  rnd_.seed(seed_++);

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}

Status PKSamplerRT::HandshakeRandomAccessOp(const RandomAccessOp *op) {
  RETURN_UNEXPECTED_IF_NULL(op);
  RETURN_IF_NOT_OK(op->GetClassIds(&label_to_ids_));
  RETURN_IF_NOT_OK(InitSampler());
  return Status::OK();
}

void PKSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: PKSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}

Status PKSamplerRT::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerRT::to_json(&args));
  args["sampler_name"] = "PKSampler";
  args["num_val"] = samples_per_class_;
  args["shuffle"] = shuffle_;
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
