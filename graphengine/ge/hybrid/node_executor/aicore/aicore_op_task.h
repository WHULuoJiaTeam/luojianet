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

#ifndef GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
#define GE_HYBRID_KERNEL_AICORE_OP_TASK_H_

#include <memory>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "runtime/stream.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/node_executor/task_context.h"
#include "proto/task.pb.h"
#include "register/op_tiling.h"

namespace ge {
namespace hybrid {
class TbeHandleHolder {
 public:
  explicit TbeHandleHolder(void *bin_handle);
  ~TbeHandleHolder();

  void SetBinHandle(void *bin_handle) { bin_handle_ = bin_handle; }
  void *GetBinHandle() { return bin_handle_; }

 private:
  friend class TbeHandleRegistry;
  void *bin_handle_ = nullptr;
};

class TbeHandleRegistry {
 public:
  static TbeHandleRegistry &GetInstance() {
    static TbeHandleRegistry instance;
    return instance;
  }

  bool AddHandle(std::unique_ptr<TbeHandleHolder> &&holder);

 private:
  std::set<std::unique_ptr<TbeHandleHolder>> registered_handles_;
};

class AiCoreOpTask {
 public:
  AiCoreOpTask() = default;
  virtual ~AiCoreOpTask() = default;

  virtual Status Init(const OpDesc &op_desc, const domi::TaskDef &task_def);

  bool IsDynamicShapeSupported();

  // do preparation with shape(without actual io memory)
  Status PrepareWithShape(TaskContext &context);

  virtual Status UpdateArgs(TaskContext &task_context);

  Status LaunchKernel(rtStream_t stream);

  const std::string& GetName() const;

  const std::string& GetLogName() const {return log_name_;}

  bool GetClearAtomic() const {return clear_atomic_;}

  uint32_t GetBlockDim() const {return block_dim_;}

  void SetSingleOp(bool is_single_op) {is_single_op_ = is_single_op;};

  virtual const std::string& GetOpType() const;

 protected:
  Status UpdateTilingInfo(TaskContext &context);
  virtual std::string GetKeyForOpParamSize() const;
  virtual std::string GetKeyForTbeKernel() const;
  virtual std::string GetKeyForTvmMagic() const;
  virtual std::string GetKeyForTvmMetaData() const;
  virtual std::string GetKeyForKernelName(const OpDesc &op_desc) const;
  virtual Status CalcTilingInfo(const NodePtr &node, optiling::utils::OpRunInfo &tiling_info);

  std::unique_ptr<TensorBuffer> tiling_buffer_ = nullptr;
  std::string tiling_data_;
  uintptr_t *arg_base_ = nullptr;
  uint32_t max_arg_count_ = 0;

 private:
  static Status ValidateTaskDef(const domi::TaskDef &task_def);
  Status InitWithTaskDef(const OpDesc &node, const domi::TaskDef &task_def);
  Status InitTilingInfo(const OpDesc &op_desc);
  Status RegisterTbeHandle(const OpDesc &op_desc);
  Status RegisterKernelHandle(const OpDesc &op_desc);
  Status InitWithKernelDef(const OpDesc &op_desc, const domi::TaskDef &task_def);
  Status InitWithKernelDefWithHandle(const OpDesc &node, const domi::TaskDef &task_def);

  std::string stub_name_;
  void *stub_func_ = nullptr;
  std::unique_ptr<uint8_t[]> args_ = nullptr;
  uint32_t args_size_ = 0;
  uint32_t block_dim_ = 1;
  bool clear_atomic_ = true;
  bool is_single_op_ = false;
  std::vector<int> output_indices_to_skip_;
  string original_kernel_key_;
  string node_info_;
  uint32_t tiling_key_ = 0;
  void *handle_ = nullptr;
  bool is_dynamic_ = false;
  uint64_t log_id_ = 0;
  std::string log_name_;
  uint32_t offset_ = 0;
  std::string op_type_;
};

class AtomicAddrCleanOpTask : public AiCoreOpTask {
 public:
  Status Init(const OpDesc &op_desc, const domi::TaskDef &task_def) override;
  Status UpdateArgs(TaskContext &task_context) override;
  const std::string& GetOpType() const override;

 protected:
  std::string GetKeyForOpParamSize() const override;
  std::string GetKeyForTbeKernel() const override;
  std::string GetKeyForTvmMagic() const override;
  std::string GetKeyForTvmMetaData() const override;
  std::string GetKeyForKernelName(const OpDesc &op_desc) const override;
  Status CalcTilingInfo(const NodePtr &node, optiling::utils::OpRunInfo &tiling_info) override;

 private:
  Status InitAtomicAddrCleanIndices(const OpDesc &op_desc);
  std::vector<int> atomic_output_indices_;
  std::vector<int> atomic_workspace_indices_;
};
}  // namespace hybrid
}  // namespace ge
#endif //GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
