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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info.h"

namespace ge {
class KernelTaskInfo : public TaskInfo {
 public:
  friend class DavinciModel;

  KernelTaskInfo()
      : ctx_(),
        fusion_op_info_(),
        stub_func_(nullptr),
        args_(nullptr),
        sm_desc_(nullptr),
        flowtable_(nullptr),
        block_dim_(0),
        args_size_(0),
        task_id_(0),
        stream_id_(0),
        so_name_(""),
        kernel_name_(""),
        kernel_type_(ccKernelType::CCE_AI_CORE),
        dump_flag_(RT_KERNEL_DEFAULT),
        dump_args_(nullptr),
        op_desc_(nullptr),
        davinci_model_(nullptr),
        skt_id_(0),
        stub_func_name_(""),
        is_l1_fusion_enable_(false),
        is_n_batch_spilt_(false),
        group_key_(-1),
        has_group_key_(false) {}

  ~KernelTaskInfo() override {
    davinci_model_ = nullptr;
    stub_func_ = nullptr;
    sm_desc_ = nullptr;
    flowtable_ = nullptr;
    args_ = nullptr;
    superkernel_device_args_addr_ = nullptr;
    superkernel_dev_nav_table_ = nullptr;
  }

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status UpdateArgs() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Release() override;

  ccOpContext *GetCtx() override { return &ctx_; }

  FusionOpInfo *GetFusionOpInfo() override { return &fusion_op_info_; }

  uint32_t GetTaskID() override { return task_id_; }

  uint32_t GetStreamId() override { return stream_id_; }

  uintptr_t GetDumpArgs() override {
    auto ret = reinterpret_cast<uintptr_t>(dump_args_);
    return ret;
  }

  uint32_t GetSktTaskID() override { return skt_id_; }

  bool CallSaveDumpInfo() override  { return call_save_dump_; };

  ccOpContext ctx_;
  FusionOpInfo fusion_op_info_;

 private:
  Status InitTVMTask(uint16_t offset, const domi::KernelDef &kernel_def);

  Status InitAICPUCustomTask(uint32_t op_index, const domi::KernelDef &kernel_def);

  Status InitCceTask(const domi::KernelDef &kernel_def);

  Status InitAicpuTask(uint32_t op_index, const domi::KernelDef &kernel_def);

  Status InitAicpuTaskExtInfo(const std::string &ext_info);

  Status StoreInputOutputTensor(const std::vector<void *> &input_data_addrs,
                                const std::vector<void *> &output_data_addrs,
                                const std::vector<::tagCcAICPUTensor> &input_descs,
                                const std::vector<::tagCcAICPUTensor> &output_descs);

  Status SetContext(const domi::KernelDef &kernel_def);

  Status UpdateCceArgs(std::string &sm_desc, std::string &flowtable, const domi::KernelDef &kernel_def);
  Status CceUpdateKernelArgs(const domi::KernelContext &context, uint64_t &data_base_addr,
                             uint64_t &weight_base_addr, uint64_t &var_base_addr, std::string &sm_desc,
                             std::string &flowtable, const domi::KernelDef &kernel_def);

  Status SetFlowtable(std::string &flowtable, const domi::KernelDef &kernel_def);

  Status UpdateL2Data(const domi::KernelDef &kernel_def);

  uint8_t IsL2CpToDDR(uint8_t origain_L2_load_to_ddr);

  static void FreeRtMem(void **ptr);

  Status SuperKernelDistribute();
  bool IsL1FusionOp(const OpDescPtr &op_desc);
  void SetIoAddrs(const OpDescPtr &op_desc);
  void InitDumpFlag();
  void InitDumpArgs(uint32_t offset);
  void SetContinuousArgs(uint32_t args_size, DavinciModel *davinci_model);
  void SetNoncontinuousArgs(uint32_t args_size, DavinciModel *davinci_model);
  Status CopyNoncontinuousArgs(uint16_t offset);

  // For super kernel
  Status SaveSKTDumpInfo();
  void UpdateTaskId();
  void UpdateSKTTaskId();
  Status SKTFinalize();
  Status SuperKernelLaunch();
  uint32_t GetDumpFlag();
  Status SaveSuperKernelInfo();
  bool IsMarkedLastNode();
  bool IsMarkedFirstNode();
  bool FirstCallSKTLaunchCheck();
  bool DoubleCallSKTSaveCheck();
  void SetArgs();

  // for blocking aicpu op
  Status DistributeWaitTaskForAicpuBlockingOp();
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support);
  Status UpdateEventIdForAicpuBlockingOp(std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> &ext_handle);

  void *stub_func_;
  void *args_;
  void *sm_desc_;
  void *flowtable_;
  uint32_t block_dim_;
  uint32_t args_size_;
  uint32_t task_id_;
  uint32_t stream_id_;
  std::string so_name_;
  std::string kernel_name_;
  ccKernelType kernel_type_;
  uint32_t dump_flag_;
  void *dump_args_;
  OpDescPtr op_desc_;   // Clear after distribute.
  vector<void *> io_addrs_;
  DavinciModel *davinci_model_;
  uint32_t args_offset_ = 0;
  uint32_t hybrid_args_offset_ = 0;
  int64_t fixed_addr_offset_ = 0;
  std::unique_ptr<uint8_t[]> args_addr = nullptr;
  uint16_t io_addr_offset_ = 0;
  bool l2_buffer_on_ = false;
  bool call_save_dump_ = false;
  int32_t topic_type_flag_ = -1;

  // aicpu ext_info device mem
  void *aicpu_ext_info_addr_ = nullptr;

  // For super kernel
  void *skt_dump_args_ = nullptr;
  uint32_t skt_id_;
  std::string stub_func_name_;
  bool is_l1_fusion_enable_;
  bool is_n_batch_spilt_;
  int64_t group_key_;
  bool has_group_key_;
  uint32_t skt_dump_flag_ = RT_KERNEL_DEFAULT;
  void *superkernel_device_args_addr_ = nullptr;
  void *superkernel_dev_nav_table_ = nullptr;
  bool is_blocking_aicpu_op_ = false;

  struct AICPUCustomInfo {
    void *input_descs = nullptr;
    void *input_addrs = nullptr;
    void *output_descs = nullptr;
    void *output_addrs = nullptr;
    void *attr_handle = nullptr;
  } custom_info_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_
