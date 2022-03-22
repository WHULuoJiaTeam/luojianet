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

#ifndef GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_

#include "external/graph/types.h"
#include "cce/aicpu_engine_struct.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info.h"

namespace ge {
namespace hybrid {
class AicpuNodeTaskBase : public NodeTask {
 public:
  AicpuNodeTaskBase(const NodeItem *node_item, const domi::TaskDef &task_def)
      : node_item_(node_item), task_def_(task_def),
        node_name_(node_item->node_name), node_type_(node_item->node_type),
        unknown_type_(node_item->shape_inference_type),
        aicpu_ext_handle_(node_item->node_name,
                          node_item->num_inputs,
                          node_item->num_outputs,
                          node_item->shape_inference_type) {}

  ~AicpuNodeTaskBase() override;

  using NodeTask::Init;

  virtual Status Init(const HybridModel &model) = 0;

  Status UpdateArgs(TaskContext &context) override;

  Status ExecuteAsync(TaskContext &context, std::function<void()> done_callback) override;
 protected:
  virtual Status InitExtInfo(const std::string &kernel_ext_info, int64_t session_id);

  virtual Status UpdateExtInfo();

  virtual Status UpdateOutputShapeFromExtInfo(TaskContext &task_context);

  Status UpdateShapeToOutputDesc(TaskContext &task_context, const GeShape &shape_new, int32_t output_index);

  virtual Status LaunchTask(TaskContext &context) = 0;

  virtual Status TaskCallback(TaskContext &context) = 0;

  virtual Status UpdateIoAddr(TaskContext &context) = 0;

  static Status AllocTensorBuffer(size_t size, std::unique_ptr<TensorBuffer> &tensor_buffer);

  Status DistributeWaitTaskForAicpuBlockingOp(rtStream_t stream);
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support);
  Status UpdateEventIdForBlockingAicpuOp();

 protected:
  const NodeItem *node_item_;
  // just reference.
  const domi::TaskDef &task_def_;

  const std::string node_name_;

  const std::string node_type_;

  // valid when node_item_->is_dynamic is true
  UnknowShapeOpType unknown_type_ = DEPEND_IN_SHAPE;

  // valid when node_item_->is_dynamic is true
  AicpuExtInfoHandler aicpu_ext_handle_;

  // ext info addr, device mem
  std::unique_ptr<TensorBuffer> ext_info_addr_dev_;

  // for blocking aicpu op
  bool is_blocking_aicpu_op_ = false;
  rtEvent_t rt_event_ = nullptr;
};

class AicpuTfNodeTask : public AicpuNodeTaskBase {
 public:
  AicpuTfNodeTask(const NodeItem *node_item, const domi::TaskDef &task_def)
      : AicpuNodeTaskBase(node_item, task_def) {}

  ~AicpuTfNodeTask() override = default;

  Status Init(const HybridModel &model) override;

 protected:

  Status LaunchTask(TaskContext &context) override;

  Status TaskCallback(TaskContext &context) override;

  Status UpdateIoAddr(TaskContext &context) override;

 private:
  Status SetMemCopyTask(const domi::TaskDef &task_def);

  Status InitForDependComputeTask();

  Status UpdateShapeAndDataByResultSummary(TaskContext &context);

  ///
  /// read result summary and prepare copy task memory.
  /// @param context task context
  /// @param out_shape_hbm if scalar, TensorBuffer->data is null, size=0
  /// @return SUCCESS:success other:failed
  ///
  Status ReadResultSummaryAndPrepareMemory(TaskContext &context,
                                           std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);
  Status CopyDataToHbm(TaskContext &context,
                       const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  Status UpdateShapeByHbmBuffer(TaskContext &context,
                                const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  Status PrepareCopyInputs(const TaskContext &context,
                           const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  static Status EnsureSessionCreated(uint64_t session_id);
  static uint64_t GetStepIdAddr(const HybridModel &model);
 private:
  // kernel buf, device mem
  std::unique_ptr<TensorBuffer> kernel_buf_;

  std::unique_ptr<TensorBuffer> kernel_workspace_;

  // input and output addr, device mem
  std::unique_ptr<TensorBuffer> input_output_addr_;

  // just used for depend DEPEND_COMPUTE op
  std::unique_ptr<TensorBuffer> copy_task_args_buf_;

  std::vector<std::unique_ptr<TensorBuffer>> output_summary_;
  std::vector<aicpu::FWKAdapter::ResultSummary> output_summary_host_;

  std::unique_ptr<TensorBuffer> copy_ioaddr_dev_;

  std::unique_ptr<TensorBuffer> copy_input_release_flag_dev_;
  std::unique_ptr<TensorBuffer> copy_input_data_size_dev_;
  std::unique_ptr<TensorBuffer> copy_input_src_dev_;
  std::unique_ptr<TensorBuffer> copy_input_dst_dev_;
  bool need_sync_ = false;

  std::unique_ptr<TensorBuffer> copy_workspace_buf_;
};

class AicpuNodeTask : public AicpuNodeTaskBase {
 public:
  AicpuNodeTask(const NodeItem *node_item, const domi::TaskDef &task_def)
      : AicpuNodeTaskBase(node_item, task_def) {}

  ~AicpuNodeTask() override = default;

  Status Init(const HybridModel &model) override;

 protected:

  Status LaunchTask(TaskContext &context) override;

  Status TaskCallback(TaskContext &context) override;

  Status UpdateIoAddr(TaskContext &context) override;

 protected:
  // host mem
  std::unique_ptr<uint8_t[]> args_;

  // args size
  uint32_t args_size_ = 0;
};

class AiCpuNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model,
                  const NodePtr &node,
                  std::shared_ptr<NodeTask> &task) const override;

  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
};
}
}
#endif //GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_
