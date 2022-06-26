/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_MAP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_MAP_OP_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/api/python/python_mp.h"
#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/map_op/map_job.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
// Forward declare
class ExecutionTree;

// A unit of job for map worker thread.
// MapWorkerJob holds a list of MapJob where each MapJob can be a CpuMapJob, GpuMapJob or DvppMapJob.
struct MapWorkerJob {
  explicit MapWorkerJob(TensorRow tr) : tensor_row(std::move(tr)) {}
  std::vector<std::shared_ptr<MapJob>> jobs;
  TensorRow tensor_row;
};

// MapOp class implements the Map operator. It will apply a list of operations to each record specified by column names.
// The column order behavior after MapOp is as follows.
// [Case 1] If the number of Input Columns == the number of Output Column, column ordering after MapOp
// is the same as the original column order where the Remainder Columns stay in the same position,
// and the Output Columns are placed the same position of the Input Columns.
// For example, initially if the dataset has column order |A, B, C, D, E|,
// and we apply MapOp() with Input Columns {B, C} and Output Columns {X, Y}.
// The column order after applying MapOp will be |A, X, Y, D, E|.
// Note that in this case, |X, Y| is the Output Columns and |A, D, E| which is the Remainder Columns stay in
// their original position, and column B is replaced by column X and column C is replace by column Y.
// [Case 2] If the number of Input Columns != the number of Output Column, column ordering after MapOp
// is Output Columns followed by Remainder Columns.
// For example, initially if the dataset has column order |A, B, C, D, E|,
// and we apply MapOp() with Input Columns {B, C, A} and Output Columns {X, Y}.
// The column order after applying MapOp will be |X, Y, D, E|.
// Note that in this case, |X, Y| is the Output Columns and |D, E| is the Remainder Columns,
// and the Input Columns are gone and replaced by the Output Columns.

// Keywords:
// Input Columns : a vector of column names (string) passed to MapOp specifying the column names from which
//     Tensors are taken and passed to the TensorOp Compute().
// Output Columns : a vector of column names (string) passed to MapOp specifying what are the column names
//     for the Tensors produced by TensorOp Compute().
// Remainder Columns : columns that exist in the dataset but are not mentioned in Input Columns.
//     These columns will not be passed to TensorOp Compute(), but will be appended to the end of the Output Columns.
class MapOp : public ParallelOp<std::unique_ptr<MapWorkerJob>, TensorRow> {
 public:
  // Constructor of MapOp
  // @note The builder class should be used to call it.
  // @param in_col_names A list of input column names (should match the input/output \p tensorFuncs).
  // @param out_col_names A list of output column names (should match the input/output \p tensorFuncs).
  // @param tensor_funcs A list of TensorOp pointers for MapOp to apply to each data.
  // @param num_workers The number of worker threads.
  // @param op_connector_size The size of each queue in the connector.
  MapOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
        std::vector<std::shared_ptr<TensorOp>> tensor_funcs, int32_t num_workers, int32_t op_connector_size);

  // Destructor
  ~MapOp() = default;

  // A print method typically used for debugging
  // @param out The output stream to write output to
  // @param show_all A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out reference to the output stream being overloaded
  // @param mo reference to the MapOp to display
  // @return the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const MapOp &mo) {
    mo.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // This main thread creates local queues, pulls TensorRow from the previous
  // op's Connector and distributes them to the local queues. Workers pull from the local queues.
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kMapOp; }

  // List of tensor ops getter/setter
  // @Return the vector of tensor ops by non-const reference

  auto &TFuncs() { return tfuncs_; }

  const auto &TFuncs() const { return tfuncs_; }

  bool IsPython() const override {
    for (const auto &tensorOp : tfuncs_) {
      if (tensorOp->Name() == kPyFuncOp) {
        return true;
      }
    }
    return false;
  }

  /// Set the instance of python multiprocessing which will passed from python
  /// \param python_mp PythonMultiprocessingRuntime
  void SetPythonMp(std::shared_ptr<PythonMultiprocessingRuntime> python_mp);

  /// Return the list of PIDs of worker processes
  /// \return vector of int
  std::vector<int32_t> GetMPWorkerPIDs() const override;

 private:
  // A helper function to create jobs for workers.
  Status GenerateWorkerJob(const std::unique_ptr<MapWorkerJob> *worker_job);

  // A helper function that fetch worker map job from local queues and extract the data and map job list
  Status FetchNextWork(uint32_t worker_id, TensorRow *row, std::vector<std::shared_ptr<MapJob>> *job_list);

  //  Tensorops to be read and applied by worker threads
  std::vector<std::shared_ptr<TensorOp>> tfuncs_;

  // Variable to store the column name that the tensorOps are consuming
  std::vector<std::string> in_columns_;

  // Variable to store the column name that the tensorOps are producing
  std::vector<std::string> out_columns_;

  // Boolean mapping, true means to keep the column.
  std::vector<bool> keep_input_columns_;

  // Indices of the columns to process.
  std::vector<size_t> to_process_indices_;

  std::unique_ptr<ChildIterator> child_iterator_;  // An iterator for fetching.

  std::shared_ptr<PythonMultiprocessingRuntime> python_mp_;  // python multiprocessing instance

  // Private function for worker/thread to loop continuously. It comprises the main
  // logic of MapOp: getting the data from previous Op, validating user specified column names,
  // applying a list of TensorOps to each of the data, process the results and then
  // pushing them back to MapOp's output Connector to be fetched by the next Op.
  // @param worker_id The id assigned to this thread/worker upon creation.
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;  //  In: workerId assigned by tree_

  // Private function for worker thread to perform TensorOp's compute function and get the result.
  // @param in_row Input TensorRow
  // @param[out] out_row Generated TensorRow
  Status WorkerCompute(const TensorRow &in_row, TensorRow *out_row,
                       const std::vector<std::shared_ptr<MapJob>> &job_list);

  // Private function that create the final column name to index mapping and
  // get indices of the columns this mapop does not use.
  // @param col_name_id_map The column name to index mapping obtained from child operator
  void CreateFinalColMap(std::unordered_map<std::string, int32_t> *col_name_id_map);

  // Validating if each of the input_columns exists in col_name_id_map.
  // @param - the column map to check
  // @return - status return code
  Status ValidateInColumns(const std::unordered_map<std::string, int32_t> &col_name_id_map);

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  // Private function for initializing private variables such as in_columns_, out_columns_.
  // @return - Status
  Status InitPrivateVariable(std::unordered_map<std::string, int32_t> *col_name_id_map);

  /// Send wait flag row to worker at worker_id to make it wait
  /// \param worker_id id of the worker
  /// \return Status code
  Status SendWaitFlagToWorker(int32_t worker_id) override;

  /// Send quit flag row to worker at worker_id to make it exit
  /// \param worker_id id of the worker
  /// \return Status code
  Status SendQuitFlagToWorker(int32_t worker_id) override;

 protected:
  Status Launch() override;

 protected:
  Status AddNewWorkers(int32_t num_new_workers) override;
  Status RemoveWorkers(int32_t num_workers) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_MAP_OP_H_
