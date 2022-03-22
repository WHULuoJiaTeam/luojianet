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

#ifndef GE_SINGLE_OP_TASK_OP_TASK_H_
#define GE_SINGLE_OP_TASK_OP_TASK_H_

#include <memory>
#include <string>
#include <external/graph/tensor.h>

#include "common/dump/dump_op.h"
#include "common/dump/dump_properties.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/op_kernel_bin.h"
#include "runtime/stream.h"
#include "graph/node.h"
#include "cce/aicpu_engine_struct.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info.h"
#include "init/gelib.h"
#include "register/op_tiling.h"

namespace ge {
namespace {
const int kAddressNum = 2;
}  // namespace

class StreamResource;
struct SingleOpModelParam;
class OpTask {
 public:
  OpTask() = default;
  virtual ~OpTask() = default;
  virtual Status LaunchKernel(rtStream_t stream) = 0;
  virtual Status UpdateRunInfo();
  virtual Status UpdateArgTable(const SingleOpModelParam &param);
  void SetModelArgs(std::string model_name, uint32_t model_id);
  Status GetProfilingArgs(TaskDescInfo &task_desc_info, uint32_t &model_id);
  const std::string &GetTaskName() const {return task_name_;}
  void SetOpDesc(const OpDescPtr &op_desc) {
    op_desc_ = op_desc;
  }
  const OpDescPtr &GetOpdesc() const {return op_desc_;}
  Status OpenDump(rtStream_t stream);
  virtual void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) = 0;
  virtual Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                              const std::vector<DataBuffer> &input_buffers,
                              std::vector<GeTensorDesc> &output_desc,
                              std::vector<DataBuffer> &output_buffers,
                              rtStream_t stream);
  virtual const std::string &GetTaskType() const;

 protected:
  Status DoUpdateArgTable(const SingleOpModelParam &param, bool keep_workspace);

  DumpProperties dump_properties_;
  DumpOp dump_op_;
  OpDescPtr op_desc_;
  std::string model_name_;
  uint32_t model_id_ = 0;
  uint32_t block_dim_ = 1;
  std::string task_name_;
};

class TbeOpTask : public OpTask {
 public:
  ~TbeOpTask() override;
  Status LaunchKernel(rtStream_t stream) override;
  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &output_buffers,
                      rtStream_t stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
  void SetSmDesc(void *sm_desc);
  void SetStubFunc(const std::string &name, const void *stub_func);
  void SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, size_t arg_size, uint32_t block_dim, const OpDescPtr &op_desc);
  void SetKernelWithHandleArgs(std::unique_ptr<uint8_t[]> &&args, size_t arg_size, uint32_t block_dim,
                               const OpDescPtr &op_desc, const domi::KernelDefWithHandle& kernel_def_with_handle);
  void SetAtomicAddrCleanTask(OpTask *task) { atomic_task_.reset(task); }

  Status UpdateRunInfo() override;
  Status SetArgIndex();

  const void *GetArgs() const;
  size_t GetArgSize() const;
  const std::string &GetStubName() const;
  Status EnableDynamicSupport(const NodePtr &node, void *tiling_buffer, uint32_t max_tiling_size);
  const std::string &GetTaskType() const override;
  void SetHandle(void *handle);

 protected:
  NodePtr node_;
  std::unique_ptr<uint8_t[]> args_;
  size_t arg_size_ = 0;
  void *tiling_buffer_ = nullptr;
  uint32_t max_tiling_size_ = 0;
  std::string tiling_data_;
  size_t input_num_; // include const input
  size_t output_num_;

 private:
  friend class SingleOpModel;
  friend class TbeTaskBuilder;
  static Status UpdateTensorDesc(const GeTensorDesc &src_tensor, GeTensorDesc &dst_tensor);
  Status AllocateWorkspaces(const std::vector<int64_t> &workspace_sizes);
  Status DoLaunchKernel(rtStream_t stream);
  Status CheckAndExecuteAtomic(const vector<GeTensorDesc> &input_desc,
                               const vector<DataBuffer> &input_buffers,
                               vector<GeTensorDesc> &output_desc,
                               vector<DataBuffer> &output_buffers,
                               rtStream_t stream);
  virtual Status UpdateNodeByShape(const vector<GeTensorDesc> &input_desc,
                                   const vector<GeTensorDesc> &output_desc);
  virtual Status UpdateTilingArgs(rtStream_t stream);
  virtual Status UpdateIoAddr(const vector<DataBuffer> &inputs, const vector<DataBuffer> &outputs);
  virtual Status CalcTilingInfo(optiling::utils::OpRunInfo &run_info);

  const void *stub_func_ = nullptr;
  void *sm_desc_ = nullptr;
  std::string stub_name_;
  StreamResource *stream_resource_ = nullptr;

  std::vector<int64_t> run_info_workspaces_;
  std::vector<void *> workspaces_;

  uint32_t tiling_key_ = 0;
  bool clear_atomic_ = false;
  void* handle_ = nullptr;
  std::string original_kernel_key_;
  std::string node_info_;
  std::vector<size_t> arg_index_; // data index in args

  std::unique_ptr<OpTask> atomic_task_;
};

class AtomicAddrCleanOpTask : public TbeOpTask {
 public:
  Status InitAtomicAddrCleanIndices();

 private:
  Status UpdateNodeByShape(const vector<GeTensorDesc> &input_desc,
                           const vector<GeTensorDesc> &output_desc) override;
  Status UpdateIoAddr(const vector<DataBuffer> &inputs, const vector<DataBuffer> &outputs) override;
  Status UpdateTilingArgs(rtStream_t stream) override;
  Status CalcTilingInfo(optiling::utils::OpRunInfo &run_info) override;
  std::vector<int> atomic_output_indices_;

};

class AiCpuBaseTask : public OpTask {
 public:
  AiCpuBaseTask() = default;
  ~AiCpuBaseTask() override;
  UnknowShapeOpType GetUnknownType() const { return unknown_type_; }
  Status UpdateArgTable(const SingleOpModelParam &param) override;
  const std::string &GetTaskType() const override;

 protected:
  Status UpdateIoAddr(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  Status SetInputConst();
  Status SetExtInfoAndType(const std::string &kernel_ext_info, uint64_t kernel_id);

  Status UpdateExtInfo(const std::vector<GeTensorDesc> &input_desc,
                       std::vector<GeTensorDesc> &output_desc,
                       rtStream_t stream);
  Status UpdateOutputShape(vector<GeTensorDesc> &output_desc);
  Status UpdateShapeToOutputDesc(const GeShape &shape_new, GeTensorDesc &output_desc);
  // for blocking aicpu op
  Status DistributeWaitTaskForAicpuBlockingOp(rtStream_t stream);
  Status UpdateEventIdForBlockingAicpuOp();
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support);

 protected:
  size_t num_inputs_ = 0;
  size_t num_outputs_ = 0;
  UnknowShapeOpType unknown_type_ = DEPEND_IN_SHAPE;
  std::unique_ptr<ge::hybrid::AicpuExtInfoHandler> aicpu_ext_handle_;
  void *ext_info_addr_dev_ = nullptr;
  vector<bool> input_is_const_;
  // for blocking aicpu op
  bool is_blocking_aicpu_op_ = false;
  rtEvent_t rt_event_ = nullptr;
};

class AiCpuTask : public AiCpuBaseTask {
 public:
  AiCpuTask() = default;
  ~AiCpuTask() override;

  Status LaunchKernel(rtStream_t stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;

  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &output_buffers,
                      rtStream_t stream) override;
  Status SetMemCopyTask(const domi::KernelExDef &kernel_def);

 private:
  // for copy task.
  Status InitForSummaryAndCopy();
  Status UpdateShapeAndDataByResultSummary(vector<GeTensorDesc> &output_desc,
                                           vector<DataBuffer> &outputs,
                                           rtStream_t stream);
  Status ReadResultSummaryAndPrepareMemory();

  Status CopyDataToHbm(vector<DataBuffer> &outputs, rtStream_t stream);
  Status PrepareCopyInputs(vector<DataBuffer> &outputs);

  Status UpdateShapeByHbmBuffer(vector<GeTensorDesc> &output_desc);

  friend class AiCpuTaskBuilder;
  void *workspace_addr_ = nullptr;
  std::string task_info_;
  // device addr
  void *args_ = nullptr;
  size_t arg_size_ = 0;
  std::string op_type_;
  // device addr
  void *io_addr_ = nullptr;
  size_t io_addr_size_ = 0;

  // host addr
  std::vector<void *> io_addr_host_;

  // for copy task
  void *copy_task_args_buf_ = nullptr;
  void *copy_workspace_buf_ = nullptr;

  std::vector<void *> output_summary_;
  std::vector<aicpu::FWKAdapter::ResultSummary> output_summary_host_;

  void *copy_ioaddr_dev_ = nullptr;

  void *copy_input_release_flag_dev_ = nullptr;
  void *copy_input_data_size_dev_ = nullptr;
  void *copy_input_src_dev_ = nullptr;
  void *copy_input_dst_dev_ = nullptr;

  vector<void *> out_shape_hbm_;
  uint64_t kernel_id_ = 0;
};

class AiCpuCCTask : public AiCpuBaseTask {
 public:
  AiCpuCCTask() = default;
  ~AiCpuCCTask() override;
  AiCpuCCTask(const AiCpuCCTask &) = delete;
  AiCpuCCTask &operator=(const AiCpuCCTask &) = delete;

  Status LaunchKernel(rtStream_t stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
  const void *GetArgs() const;
  void SetKernelArgs(std::unique_ptr<uint8_t[]> args, size_t arg_size);
  void SetSoName(const std::string &so_name);
  void SetkernelName(const std::string &kernel_Name);
  void SetIoAddr(uintptr_t *io_addr);
  size_t GetArgSize() const;

  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &output_buffers,
                      rtStream_t stream)  override;

private:
  friend class AiCpuCCTaskBuilder;
  std::string so_name_;
  std::string kernel_name_;
  std::unique_ptr<uint8_t[]> args_;
  size_t arg_size_ = 0;
  void *sm_desc_ = nullptr;
  uintptr_t *io_addr_ = nullptr;
  size_t io_addr_num_ = 0;
  bool is_custom_ = false;
  uint32_t dump_flag_ = RT_KERNEL_DEFAULT;
  std::string op_type_;
  uint64_t kernel_id_ = 0;
};

class MemcpyAsyncTask : public OpTask {
 public:
  Status LaunchKernel(rtStream_t stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;

 private:
  friend class SingleOpModel;
  friend class RtsKernelTaskBuilder;

  uintptr_t addresses_[kAddressNum] = {0};
  size_t dst_max_;
  size_t count_;
  rtMemcpyKind_t kind_;
  NodePtr node_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_OP_TASK_H_
