/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_OPERATOR_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_OPERATOR_H_

#include <memory>
#include <vector>
#include "minddata/mindrecord/include/shard_task_list.h"
#include "minddata/dataset/include/dataset/constants.h"

namespace mindspore {
namespace mindrecord {
class __attribute__((visibility("default"))) ShardOperator {
 public:
  virtual ~ShardOperator() = default;

  Status operator()(ShardTaskList &tasks) {
    RETURN_IF_NOT_OK(this->PreExecute(tasks));
    RETURN_IF_NOT_OK(this->Execute(tasks));
    RETURN_IF_NOT_OK(this->SufExecute(tasks));
    return Status::OK();
  }

  virtual bool HasChildOp() { return child_op_ != nullptr; }

  virtual Status SetChildOp(const std::shared_ptr<ShardOperator> &child_op) {
    if (child_op != nullptr) {
      child_op_ = child_op;
    }
    return Status::OK();
  }

  virtual std::shared_ptr<ShardOperator> GetChildOp() { return child_op_; }

  virtual Status PreExecute(ShardTaskList &tasks) { return Status::OK(); }

  virtual Status Execute(ShardTaskList &tasks) = 0;

  virtual Status SufExecute(ShardTaskList &tasks) { return Status::OK(); }

  /// \brief compute actual the num_samples via loading data
  virtual int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) { return 0; }

  /// \brief Getter the number of samples which is set via python api
  virtual int64_t GetNumSamples() const { return num_samples_; }

  /// \brief Setter the number of samples in python
  virtual void SetNumSamples(int64_t num_samples) { num_samples_ = num_samples; }

  virtual void UpdateShuffleMode(dataset::ShuffleMode shuffle_mode) { shuffle_mode_ = shuffle_mode; }

  virtual dataset::ShuffleMode GetShuffleMode() { return shuffle_mode_; }

  virtual void SetShardSampleCount(const std::vector<int64_t> &shard_sample_count) {
    shard_sample_count_ = shard_sample_count;
  }

  virtual std::vector<int64_t> GetShardSampleCount() { return shard_sample_count_; }

 private:
  int64_t num_samples_ = 0;
  std::shared_ptr<ShardOperator> child_op_ = nullptr;
  // indicate shard_id : inc_count
  // 0 : 15  -  shard0 has 15 samples
  // 1 : 41  -  shard1 has 26 samples
  // 2 : 58  -  shard2 has 17 samples
  std::vector<int64_t> shard_sample_count_;
  dataset::ShuffleMode shuffle_mode_ = dataset::ShuffleMode::kGlobal;
};
}  // namespace mindrecord
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_OPERATOR_H_
