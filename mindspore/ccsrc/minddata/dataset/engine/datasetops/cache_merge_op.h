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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_MERGE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_MERGE_OP_H_

#include <algorithm>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/queue_map.h"
#include "minddata/dataset/util/semaphore.h"

namespace mindspore {
namespace dataset {
/// \brief Provides method to merge two streams (one from CacheLookup and one from cache miss stream) into one single
/// stream
class CacheMergeOp : public ParallelOp<TensorRow, TensorRow> {
 public:
  // Some handshake structures between CacheMissWorkerEntry and Cleaner
  class TensorRowCacheRequest {
   public:
    enum class State : uint8_t {
      kEmpty = 0,  // Initial state. Row hasn't arrived from cache miss stream yet.
      kDirty = 1,  // Cleaner hasn't flushed it to the cache server yet.
      kClean = 2   // The row has been flushed already.
    };
    TensorRowCacheRequest() : st_(State::kEmpty) {}
    ~TensorRowCacheRequest() = default;
    /// Getter and Setter of the state
    State GetState() const { return st_; }
    void SetState(State newState) { st_ = newState; }
    /// Take a tensor row and send rpc call to the server async
    /// \param cc Cache client of the CacheMergeOp
    /// \param row TensorRow to be sent to the server
    /// \return Status object
    /// \note Thread safe
    Status AsyncSendCacheRequest(const std::shared_ptr<CacheClient> &cc, const TensorRow &row);

    /// \brief We send the row to the server async so the CacheMissWorkerEntry can continue.
    /// It is the cleaner that will check the result.
    /// \return Status object
    Status CheckCacheResult();

   private:
    std::atomic<State> st_;
    std::shared_ptr<CacheRowRequest> cleaner_copy_;
  };

  constexpr static int kNumChildren = 2;        // CacheMergeOp has 2 children
  constexpr static int kCacheHitChildIdx = 0;   // Cache hit stream
  constexpr static int kCacheMissChildIdx = 1;  // Cache miss stream

  /// \brief Constructor
  /// \param numWorkers Number of parallel workers as a derived class of ParallelOp
  /// \param opConnector Size Connector size as a derived class of ParallelOp
  /// \param numCleaners Number of cleaners to move cache miss rows into the cache server
  /// \param cache_client CacheClient to communicate with the Cache server
  CacheMergeOp(int32_t numWorkers, int32_t opConnectorSize, int32_t numCleaners,
               std::shared_ptr<CacheClient> cache_client);
  ~CacheMergeOp();
  void Print(std::ostream &out, bool show_all) const override;
  std::string Name() const override { return kCacheMergeOp; }

  friend std::ostream &operator<<(std::ostream &out, const CacheMergeOp &mo) {
    mo.Print(out, false);
    return out;
  }
  /// \brief Master thread responsible to spawn all the necessary worker threads for the two streams and
  ///     the threads for the cleaners.
  /// \return
  Status operator()() override;

  /// \brief Entry function for worker thread that fetch rows from CacheLookupOp
  /// \param workerId
  /// \return Status object
  Status WorkerEntry(int32_t workerId) override;

  /// \brief Perform specific post-operations on CacheOp
  /// \return Status The status code returned
  Status PrepareOperator() override;

  /// \brief Main thread to fetch rows from the miss child and assign it to workers
  /// \return Status The status code returned
  Status CacheMissMaster();

  /// \brief Entry function for worker thread that fetch rows from the cache miss stream
  /// \param workerId
  /// \return Status object
  Status CacheMissWorkerEntry(int32_t workerId);

  /// \brief Base-class override for eoe handling
  /// \param worker_id
  /// \return Status object
  Status EoeReceived(int32_t worker_id) override;

  /// \brief Base-class override for handling cases when an eof is received.
  /// \param worker_id - The worker id
  /// \return Status The status code returned
  Status EofReceived(int32_t worker_id) override;

 protected:
  Status ComputeColMap() override;

 private:
  std::mutex mux_;
  QueueMap<row_id_type, TensorRow> cache_miss_;
  std::map<row_id_type, MemGuard<TensorRowCacheRequest, Allocator<TensorRowCacheRequest>>> io_request_;
  std::unique_ptr<Queue<row_id_type>> io_que_;
  int32_t num_cleaners_;
  std::shared_ptr<CacheClient> cache_client_;
  std::atomic<bool> cache_missing_rows_;

  QueueList<TensorRow> missWorkers_in_queues_;

  /// \brief Locate the cache request from the io_request_ map
  /// \param row_id
  /// \param out pointer to the cache request
  /// \return Status object
  Status GetRq(row_id_type row_id, TensorRowCacheRequest **out);

  /// \brief These are the entry functions for the cleaner threads. Each cleaner is responsible for
  /// moving cache miss TensorRow into the CacheServer.
  /// \return Status object
  Status Cleaner();
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_CACHE_MERGE_OP_H_
