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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MAPPABLE_LEAF_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MAPPABLE_LEAF_OP_H_

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

class MappableLeafOp : public ParallelOp<std::unique_ptr<IOBlock>, TensorRow>, public RandomAccessOp {
 public:
  /// Constructor
  /// \param int32_t num_wkrs - Num of workers reading images in parallel
  /// \param int32_t queue_size - connector queue size
  /// \param td::unique_ptr<Sampler> sampler - sampler tells the source  what to read
  MappableLeafOp(int32_t num_wkrs, int32_t queue_size, std::shared_ptr<SamplerRT> sampler);

  /// Destructor.
  ~MappableLeafOp() = default;

  /// Main Loop of MappableLeaf
  /// Master thread: Fill IOBlockQueue, then goes to sleep
  /// Worker thread: pulls IOBlock from IOBlockQueue, work on it then put row to out_connector_
  /// \return Status The status code returned
  Status operator()() override;

  /// Op name getter
  /// @return Name of the current Op
  std::string Name() const override { return "MappableLeafPp"; }

 protected:
  /// Initialize Sampler, calls sampler->Init() within
  /// @return Status The status code returned
  Status InitSampler();

  virtual Status InitOp() {
    // The order of the following 2 functions must not be changed!
    RETURN_IF_NOT_OK(this->PrepareData());  // Prepare data
    RETURN_IF_NOT_OK(this->InitSampler());  // pass numRows to Sampler
    return Status::OK();
  }

  virtual Status PrepareData() = 0;

  /// Worker thread pulls a number of IOBlock from IOBlock Queue, make a row and push it to Connector
  /// \param int32_t workerId - id of each worker
  /// \return Status The status code returned
  Status WorkerEntry(int32_t workerId) override;

  /// Virtual function to Load a tensor row at location row_id
  /// \param row_id_type row_id - id for this tensor row
  /// \param TensorRow row - loaded row
  /// \return Status The status code returned
  virtual Status LoadTensorRow(row_id_type row_id, TensorRow *row) = 0;

  /// Reset function to be called after every epoch to reset the source op after
  /// \return Status The status code returned
  Status Reset() override;
  Status SendWaitFlagToWorker(int32_t worker_id) override;
  Status SendQuitFlagToWorker(int32_t worker_id) override;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_MAPPABLE_LEAF_OP_H_
