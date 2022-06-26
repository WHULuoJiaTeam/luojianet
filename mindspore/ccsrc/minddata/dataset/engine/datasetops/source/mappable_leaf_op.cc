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
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
MappableLeafOp::MappableLeafOp(int32_t num_wkrs, int32_t queue_size, std::shared_ptr<SamplerRT> sampler)
    : ParallelOp(num_wkrs, queue_size, std::move(sampler)) {}

// Main logic, Register Queue with TaskGroup, launch all threads and do the functor's work
Status MappableLeafOp::operator()() {
  // Registering and launching worker threads have to be before in sync with caller (i.e., before FindMe()::Post())
  RETURN_IF_NOT_OK(RegisterAndLaunchThreads());
  // Initialize callback
  RETURN_IF_NOT_OK(callback_manager_.Init(this));
  // Synchronize with TaskManager
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(InitOp());

  int64_t ep_step = 0, total_step = 0;
  RETURN_IF_NOT_OK(callback_manager_.Begin(CallbackParam(0, ep_step, total_step)));
  TensorRow sample_row;
  RETURN_IF_NOT_OK(sampler_->GetNextSample(&sample_row));
  while (true) {  // each iteration is 1 epoch, breaks when IsLastIteration() is true
    if (op_current_repeats_ % GetOpNumRepeatsPerEpoch() == 0) {
      ep_step = 0;
      RETURN_IF_NOT_OK(callback_manager_.EpochBegin(CallbackParam(op_current_epochs_ + 1, ep_step, total_step)));
    }
    while (sample_row.eoe() == false) {
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
        if ((*itr) >= num_rows_) {
          MS_LOG(WARNING) << "Skipping sample with ID: " << *itr << " since it is out of bound: " << num_rows_;
          continue;  // index out of bound, skipping
        }
        ep_step++;
        total_step++;
        RETURN_IF_NOT_OK(callback_manager_.StepBegin(CallbackParam(op_current_epochs_ + 1, ep_step, total_step)));
        RETURN_IF_NOT_OK(
          worker_in_queues_[NextWorkerID()]->Add(std::make_unique<IOBlock>(*itr, IOBlock::kDeIoBlockNone)));
      }
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sample_row));
    }
    RETURN_IF_NOT_OK(worker_in_queues_[NextWorkerID()]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    if (!IsLastIteration()) {
      // If not the last repeat, self-reset and go to loop again.
      RETURN_IF_NOT_OK(Reset());
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sample_row));
    } else {
      break;
    }
    UpdateRepeatAndEpochCounter();
  }
  RETURN_IF_NOT_OK(worker_in_queues_[NextWorkerID()]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof)));
  for (int32_t i = 0; i < num_workers_; ++i) {
    RETURN_IF_NOT_OK(SendQuitFlagToWorker(NextWorkerID()));
  }
  return Status::OK();
}

// Reset Sampler and wakeup Master thread (functor)
Status MappableLeafOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(sampler_->ResetSampler());
  return Status::OK();
}

// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status MappableLeafOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

// contains the main logic of pulling a IOBlock from IOBlockQueue, load a row and push the row to out_connector_
// IMPORTANT: 1 IOBlock produces 1 row
Status MappableLeafOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->wait()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(TensorRow(TensorRow::TensorRowFlags::kFlagWait)));
    } else if (io_block->eoe()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(TensorRow(TensorRow::TensorRowFlags::kFlagEOE)));
    } else if (io_block->eof()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(TensorRow(TensorRow::TensorRowFlags::kFlagEOF)));
    } else {
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(io_block->GetKeys(&keys));
      if (keys.empty()) return Status::OK();  // empty key is a quit signal for workers
      TensorRow trow;
      RETURN_IF_NOT_OK(this->LoadTensorRow(keys[0], &trow));
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(std::move(trow)));
    }
    RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("[Internal ERROR] Unexpected nullptr received in worker.");
}

Status MappableLeafOp::SendWaitFlagToWorker(int32_t worker_id) {
  RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagWait)));
  return Status::OK();
}

Status MappableLeafOp::SendQuitFlagToWorker(int32_t worker_id) {
  RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockNone)));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
