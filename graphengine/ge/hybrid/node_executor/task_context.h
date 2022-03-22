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

#ifndef GE_HYBRID_KERNEL_TASK_CONTEXT_H_
#define GE_HYBRID_KERNEL_TASK_CONTEXT_H_

#include <map>
#include <mutex>
#include <vector>
#include "common/properties_manager.h"
#include "external/ge/ge_api_error_codes.h"
#include "framework/common/ge_types.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {
struct GraphExecutionContext;
class SubgraphContext;

class TaskContext {
 public:
  static std::unique_ptr<TaskContext> Create(NodeState *node_state, SubgraphContext *subgraph_context);

  ~TaskContext();

  int NumInputs() const;
  int NumOutputs() const;
  size_t NumWorkspaces() const;
  const NodeItem &GetNodeItem() const;
  NodeState *GetNodeState() const;
  const char *GetNodeName() const;
  TensorValue *MutableInput(int index);
  ConstGeTensorDescPtr GetInputDesc(int index) const;
  Status GetInputDesc(int index, GeTensorDesc &tensor_desc) const;
  ConstGeTensorDescPtr GetOutputDesc(int index) const;
  Status GetOutputDesc(int index, GeTensorDesc &tensor_desc) const;
  GeTensorDescPtr MutableInputDesc(int index) const;
  GeTensorDescPtr MutableOutputDesc(int index) const;
  Status UpdateInputDesc(int index, const GeTensorDesc &tensor_desc);
  void ReleaseInputsAndOutputs();
  bool NeedCallback();
  void ReleaseInput(int index);
  void ReleaseWorkspace();
  const TensorValue *GetInput(int index) const;
  const TensorValue *GetOutput(int index) const;
  TensorValue *MutableOutput(int index);
  TensorValue *GetVariable(const std::string &name);
  rtStream_t GetStream() const;
  int64_t GetSessionId() const;
  uint64_t GetIterationNumber() const;


  void NodeDone();
  void OnError(Status error);

  Status SetOutput(int index, const TensorValue &tensor);
  Status AllocateOutput(int index,
                        const GeTensorDesc &tensor_desc,
                        TensorValue **tensor,
                        AllocationAttr *attr = nullptr);
  Status AllocateOutputs(AllocationAttr *attr = nullptr);
  Status AllocateWorkspaces();
  Status AllocateWorkspace(size_t size, void **buffer, void *ori_addr = nullptr);

  bool IsTraceEnabled() const;

  bool IsDumpEnabled() const;

  const DumpProperties& GetDumpProperties() const;

  const GraphExecutionContext *GetExecutionContext() {
    return execution_context_;
  }

  Status AllocateTensor(size_t size, TensorValue &tensor, AllocationAttr *attr = nullptr);
  void *MutableWorkspace(int index);
  const void *GetVarBaseAddr();

  Status RegisterCallback(const std::function<void()> &callback_fun) const;
  Status TryExecuteCallback(const std::function<void()> &callback_fun) const;

  Status PropagateOutputs();

  Status GetStatus() const;

  void SetStatus(Status status);

  uint32_t GetTaskId() const;
  void SetTaskId(uint32_t task_id);

  uint32_t GetStreamId() const;
  void SetStreamId(uint32_t stream_id);

  void SetOverFlow(bool is_over_flow);
  bool IsOverFlow();

  Status Synchronize();

  bool IsForceInferShape() const;
  void SetForceInferShape(bool force_infer_shape);
  void *handle_ = nullptr;

  const std::vector<TaskDescInfo>& GetProfilingTaskDescInfo() const { return task_desc_info; }
  Status SaveProfilingTaskDescInfo(uint32_t task_id, uint32_t stream_id, const std::string &task_type,
                                   uint32_t block_dim, const std::string &op_type);
  void ClearProfilingTaskDescInfo() { task_desc_info.clear(); }

 private:
  TaskContext(GraphExecutionContext *execution_context,
              NodeState *node_state,
              SubgraphContext *subgraph_context);

  static string TensorDesc2String(const GeTensorDesc &desc);
  Status AllocateTensor(const GeTensorDesc &tensor_desc, TensorValue &tensor, AllocationAttr *attr);

  NodeState *node_state_ = nullptr;
  const NodeItem *node_item_ = nullptr;
  bool force_infer_shape_ = false;
  GraphExecutionContext *execution_context_;
  SubgraphContext *subgraph_context_;
  TensorValue *inputs_start_ = nullptr;
  TensorValue *outputs_start_ = nullptr;
  Status status_ = SUCCESS;
  std::vector<void *> workspaces_;
  uint64_t iteration_ = 0;
  uint32_t task_id_ = 0;
  uint32_t stream_id_ = 0;
  std::vector<TaskDescInfo> task_desc_info;
  bool is_over_flow_ = false;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_KERNEL_TASK_CONTEXT_H_
