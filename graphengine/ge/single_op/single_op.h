/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef GE_SINGLE_OP_SINGLE_OP_H_
#define GE_SINGLE_OP_SINGLE_OP_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/executor/ge_executor.h"
#include "runtime/stream.h"
#include "single_op/task/op_task.h"
#include "cce/aicpu_engine_struct.h"
#include "hybrid/executor/hybrid_model_executor.h"

namespace ge {
class StreamResource;
struct SingleOpModelParam;
class SingleOp {
 public:
  SingleOp(StreamResource *stream_resource, std::mutex *stream_mutex, rtStream_t stream);
  ~SingleOp();

  Status ExecuteAsync(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  void SetStream(rtStream_t stream);

 private:
  Status ValidateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  Status UpdateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  Status GetArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);

  friend class SingleOpModel;
  StreamResource *stream_resource_ = nullptr;
  std::mutex *stream_mutex_;
  rtStream_t stream_ = nullptr;
  std::vector<void *> input_addr_list_;
  std::vector<size_t> input_sizes_;
  std::vector<void *> output_addr_list_;
  std::vector<size_t> output_sizes_;
  std::vector<uintptr_t> args_;

  std::vector<OpTask *> tasks_;
  std::vector<std::vector<uintptr_t *>> arg_table_;
  std::unique_ptr<SingleOpModelParam> running_param_;
  std::unique_ptr<hybrid::HybridModel> hybrid_model_;
  std::unique_ptr<hybrid::HybridModelExecutor> hybrid_model_executor_;
  std::vector<GeTensorDesc> inputs_desc_;
};

class DynamicSingleOp {
 public:
  DynamicSingleOp(uintptr_t resource_id, std::mutex *stream_mutex_, rtStream_t stream);
  ~DynamicSingleOp() = default;
  Status ExecuteAsync(const vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &inputs,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &outputs);

 private:
  friend class SingleOpModel;
  Status ValidateParams(const vector<GeTensorDesc> &input_desc,
                        const std::vector<DataBuffer> &inputs,
                        std::vector<GeTensorDesc> &output_desc,
                        std::vector<DataBuffer> &outputs) const;
  Status SetHostTensorValue(const std::vector<std::pair<size_t, uint64_t>> &inputs_size,
                            const vector<GeTensorDesc> &input_desc, const std::vector<DataBuffer> &input_buffers);
  Status SetHostTensorValue(const vector<GeTensorDesc> &input_desc, const vector<DataBuffer> &input_buffers);
  std::unique_ptr<OpTask> op_task_;
  std::unique_ptr<hybrid::HybridModel> hybrid_model_;
  std::unique_ptr<hybrid::HybridModelExecutor> hybrid_model_executor_;
  std::map<int32_t, std::vector<GeTensorDescPtr>> tensor_with_hostmem_;

  uintptr_t resource_id_ = 0;
  std::mutex *stream_mutex_;
  rtStream_t stream_ = nullptr;
  size_t num_inputs_ = 0;
  size_t num_outputs_ = 0;
  ComputeGraphPtr compute_graph_;
};
}  // namespace ge
#endif  // GE_SINGLE_OP_SINGLE_OP_H_
