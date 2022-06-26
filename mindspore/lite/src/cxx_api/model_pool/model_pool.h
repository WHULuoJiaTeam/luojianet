/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_MODEL_POOL_H_
#define MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_MODEL_POOL_H_
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include <string>
#include <queue>
#include <map>
#include "src/runtime/dynamic_mem_allocator.h"
#include "include/api/status.h"
#include "include/api/context.h"
#include "include/api/model_parallel_runner.h"
#include "src/cxx_api/model_pool/model_worker.h"
#include "src/cxx_api/model_pool/predict_task_queue.h"
namespace mindspore {
struct ModelPoolContext {
  std::shared_ptr<Context> context = nullptr;
  int numa_id = 0;
};
using ModelPoolContextVec = std::vector<std::shared_ptr<ModelPoolContext>>;

class ModelPool {
 public:
  ModelPool() = default;

  ~ModelPool();

  Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr);

  std::vector<MSTensor> GetInputs();

  std::vector<MSTensor> GetOutputs();

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

 private:
  ModelPoolContextVec CreateModelContext(const std::shared_ptr<RunnerConfig> &runner_config);
  std::shared_ptr<Context> InitContext(const std::shared_ptr<RunnerConfig> &runner_config);

  Status InitDefaultContext(const std::shared_ptr<mindspore::Context> &context);
  std::shared_ptr<Context> InitUserDefineContext(const std::shared_ptr<RunnerConfig> &runner_config);
  Status SetDefaultOptimalModelNum(const std::shared_ptr<mindspore::Context> &context);

  Status SetModelBindMode(std::vector<std::vector<int>> *all_model_bind_list, std::vector<int> *numa_node_id,
                          std::shared_ptr<Context> model_context);
  Status SetNumaBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, std::vector<int> *numa_node_id,
                             int thread_num);
  void SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, std::vector<int> *numa_node_id,
                       int thread_num);

  Status SplitInputTensorByBatch(const std::vector<MSTensor> &inputs, std::vector<std::vector<MSTensor>> *new_inputs,
                                 size_t batch_split_num);
  Status SplitOutputTensorByBatch(std::vector<std::vector<MSTensor>> *outputs, std::vector<MSTensor> *new_outputs,
                                  size_t batch_split_num);
  Status ConcatPredictOutput(std::vector<std::vector<MSTensor>> *outputs, std::vector<MSTensor> *new_outputs,
                             int numa_id);
  Status FreeSplitTensor(std::vector<std::vector<MSTensor>> *new_inputs,
                         std::vector<std::vector<MSTensor>> *new_outputs);
  void GetMaxWaitWorkerNum(int *max_wait_worker_node_id, int *max_wait_worker_num);

  std::vector<std::thread> model_worker_vec_;
  std::vector<std::shared_ptr<ModelWorker>> model_workers_;
  std::vector<MSTensor> model_inputs_;
  std::vector<MSTensor> model_outputs_;
  char *graph_buf_ = nullptr;
  size_t workers_num_ = 1;
  std::mutex mtx_split_task_;
  bool is_user_data_ = false;
  int numa_node_num_ = 1;
  int used_numa_node_num_ = 0;
  bool use_numa_bind_mode_ = false;
  bool use_gpu_ = false;
  std::shared_ptr<PredictTaskQueue> predict_task_queue_ = nullptr;
  std::unordered_map<int, std::shared_ptr<Allocator>> numa_allocator_;
  bool use_split_batch_ = false;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_MODEL_POOL_H_
