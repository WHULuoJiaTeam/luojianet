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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "framework/common/ge_types.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/helper/om_file_helper.h"
#include "common/opskernel/ge_task_info.h"
#include "common/properties_manager.h"
#include "common/dump/exception_dumper.h"
#include "common/dump/opdebug_register.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/aipp_utils.h"
#include "graph/load/model_manager/data_dumper.h"
#include "graph/load/model_manager/data_inputer.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/zero_copy_offset.h"
#include "graph/load/model_manager/zero_copy_task.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "external/graph/operator.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "mmpa/mmpa_api.h"
#include "proto/task.pb.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "common/local_context.h"

using std::mutex;
using std::thread;
using std::multimap;

namespace ge {
// op debug need 2048 bits buffer
const size_t kOpDebugMemorySize = 2048UL;
const size_t kDebugP2pSize = 8UL;

typedef enum tagModelProcStage {
  MODEL_LOAD_START = 1,
  MODEL_LOAD_END,
  MODEL_PRE_PROC_START,
  MODEL_PRE_PROC_END,
  MODEL_INFER_START,
  MODEL_INFER_END,
  MODEL_AFTER_PROC_START,
  MODEL_AFTER_PROC_END,
  MODEL_PROC_INVALID,
} ModelProcStage;

struct timeInfo {
  uint32_t modelId;
  int64_t processBeginTime;
  int64_t processEndTime;
  int64_t inferenceBeginTime;
  int64_t inferenceEndTime;
  int64_t dumpBeginTime;
  int64_t dumpEndTime;
};

// For super kernel
struct SuperKernelTaskInfo {
  uint32_t last_block_dim;
  uint32_t last_args_size;
  uint32_t last_task_id;
  uint32_t last_stream_id;
  void *last_stream;
  void *last_sm_desc;
  vector<void *> kernel_list;
  vector<void *> arg_list;
  vector<uint32_t> dump_flag_list;
  vector<OpDescPtr> op_desc_list;
  vector<uintptr_t> dump_args_list;
  uint32_t last_dump_flag;
  int64_t last_group_key;
  uintptr_t last_dump_args;
  OpDescPtr last_op;
};

struct TaskMemInfo {
  int64_t input_size{0};
  int64_t output_size{0};
  int64_t weight_size{0};
  int64_t workspace_size{0};
  int64_t total_size{0};
};

struct ProfileInfo {
  FusionOpInfo fusion_info;
  TaskMemInfo memory_info;
  uint32_t task_count{0};
};

enum ExecuteMode {
  INITIALIZATION,
  SYNCHRONIZATION,
  ASYNCHRONIZATION,
};

// comments
class DavinciModel {
 public:
  ///
  /// @ingroup ge
  /// @brief DavinciModel constructor
  /// @author
  ///
  DavinciModel(int32_t priority, const shared_ptr<ModelListener> &listener);

  ///
  /// @ingroup ge
  /// @brief DavinciModel desctructor, free Parse and Init resources
  /// @author
  ///
  ~DavinciModel();

  ///
  /// @ingroup ge
  /// @brief apply model to model_def_
  ///
  Status Assign(const GeModelPtr &ge_model);

  ///
  /// @ingroup ge
  /// @brief DavinciModel initialization, including Stream, ccHandle, Event, DataInputer, etc
  /// @return execute result
  /// @author
  ///
  Status Init(void *dev_ptr = nullptr, size_t memsize = 0, void *weight_ptr = nullptr, size_t weightsize = 0);

  ///
  /// @ingroup ge
  /// @brief ACL case, Load task list with queue.
  /// @param [in] input_que_ids: input queue ids from user, nums equal Data Op.
  /// @param [in] output_que_ids: input queue ids from user, nums equal NetOutput Op.
  /// @return: 0 for success / others for fail
  ///
  Status SetQueIds(const vector<uint32_t> &input_queue_ids, const vector<uint32_t> &output_queue_ids);

  ///
  /// @ingroup ge
  /// @brief Get DataInputer
  /// @return model ID
  ///
  uint32_t Id() const { return model_id_; }

  ///
  /// @ingroup ge
  /// @brief Get DataInputer
  /// @return model ID
  ///
  void SetId(uint32_t model_id) { model_id_ = model_id; }

  ///
  /// @ingroup ge
  /// @brief Get SubModelId
  /// @return sub model ID
  ///
  uint32_t SubModelId() const { return sub_model_id_; }

  ///
  /// @ingroup ge
  /// @brief Get SubModelId
  /// @return sub model ID
  ///
  void SetSubModelId(uint32_t sub_model_id) { sub_model_id_ = sub_model_id; }

  static void *Run(DavinciModel *model_pointer);

  ///
  /// @ingroup ge
  /// @brief NnExecute
  /// @param [in] stream   execute stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] input_data  model input data
  /// @param [out] output_data  model output data
  ///
  Status NnExecute(rtStream_t stream, bool async_mode, const InputData &input_data, OutputData &output_data);

  ///
  /// @ingroup ge
  /// @brief lock mutex run flag
  /// @author
  ///
  void LockRunFlg() { mux_run_flg_.lock(); }

  ///
  /// @ingroup ge
  /// @brief unlock mutex run flag
  /// @author
  ///
  void UnlockRunFlg() { mux_run_flg_.unlock(); }

  ///
  /// @ingroup ge
  /// @brief get DataInputer
  /// @return DataInputer pointer
  ///
  DataInputer *const GetDataInputer() const { return data_inputer_; }

  uint32_t GetDataInputerSize() {
    GE_CHECK_NOTNULL(data_inputer_);
    return data_inputer_->Size();
  }

  // get Stream number
  uint32_t StreamNum() const { return runtime_param_.stream_num; }

  // get Event number
  uint32_t EventNum() const { return runtime_param_.event_num; }

  // get Lable number
  uint32_t LabelNum() const { return runtime_param_.label_num; }

  // get batch number
  uint32_t BatchNum() const { return runtime_param_.batch_num; }

  // get session id
  uint64_t SessionId() const { return runtime_param_.session_id; }

  // get model priority
  int32_t Priority() const { return priority_; }

  // get total mem size
  size_t TotalMemSize() const { return runtime_param_.mem_size; }

  ///
  /// @ingroup ge
  /// @brief Get total useful size, in known subgraph, no need to allocate zero copy memory during initialization.
  /// @param [in] total_useful_size: total mem size - zero copy size.
  /// @return Status
  ///
  Status GetTotalMemSizeExcludeZeroCopy(int64_t &total_useful_size);

  // model name
  string Name() const { return name_; }

  // om_name
  const string &OmName() const { return om_name_; }

  // dump_model_name
  const string &DumpModelName() const { return dump_model_name_; }

  // version
  uint32_t Version() const { return version_; }

  // get total weights mem size
  size_t TotalWeightsMemSize() const { return runtime_param_.weight_size; }

  size_t TotalVarMemSize() const { return runtime_param_.var_size; }

  // get base memory address
  uint8_t *MemBase() { return mem_base_; }

  // get weight base memory address
  uint8_t *WeightsMemBase() { return weights_mem_base_; }

  uint8_t *VarMemBase() { return var_mem_base_; }

  // get Event list
  const vector<rtEvent_t> &GetEventList() const { return event_list_; }

  const vector<rtStream_t> &GetStreamList() const { return stream_list_; }

  const vector<rtLabel_t> &GetLabelList() const { return label_list_; }

  Status GetLabelGotoAddr(uint32_t label_index, rtMemType_t memory_type, void *&addr, uint32_t &size);

  Status DestroyThread();

  // get Op
  OpDescPtr GetOpByIndex(uint32_t index) const {
    if (op_list_.find(index) == op_list_.end()) {
      return nullptr;
    }
    return op_list_.at(index);
  }

  void SetGlobalStep(void *global_step, uint64_t global_step_size);
  void *GetGlobalStep() const { return global_step_addr_; }

  // get task info for profiling
  const vector<TaskDescInfo> &GetTaskDescInfo() const { return task_desc_info_; }

  // get updated task info list
  vector<TaskInfoPtr> GetTaskList() { return task_list_; }

  // Modified from KernelTaskInfo.
  SuperKernelTaskInfo &GetSuperKernelTaskInfo() { return skt_info_; }

  rtModel_t GetRtModelHandle() const { return rt_model_handle_; }

  rtStream_t GetRtModelStream() const { return rt_model_stream_; }

  uint64_t GetRtBaseAddr() const { return runtime_param_.logic_mem_base; }

  uint64_t GetRtWeightAddr() const { return runtime_param_.logic_weight_base; }

  uint64_t GetRtVarAddr() const { return runtime_param_.logic_var_base; }

  uint32_t GetFlowctrlIndex(uint32_t op_index);

  void PushHcclStream(rtStream_t value);

  bool IsBroadCastOpData(const NodePtr &var_node);

  ///
  /// @ingroup ge
  /// @brief For TVM Op, avoid Addr Reuse.
  /// @return void*
  ///
  const char *GetRegisterStub(const string &tvm_binfile_key, const string &session_graph_model_id = "");

  ///
  /// @ingroup ge
  /// @brief get model input and output desc info
  /// @param [out] input_shape  model input size
  /// @param [out] output_shape model output size
  /// @return execute result
  ///
  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc, vector<InputOutputDescInfo> &output_desc);

  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc, vector<InputOutputDescInfo> &output_desc,
                                vector<uint32_t> &input_formats, vector<uint32_t> &output_formats, bool by_dims);

  ///
  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  ///
  Status GetDynamicBatchInfo(vector<vector<int64_t>> &batch_info, int32_t &dynamic_type) const;

  ///
  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [out] batch_info
  /// @return None
  ///
  void GetCombinedDynamicDims(vector<vector<int64_t>> &batch_info) const;

  void GetUserDesignateShapeOrder(vector<string> &user_input_shape_order) const;

  void GetCurShape(vector<int64_t> &batch_info, int32_t &dynamic_type) const;

  Status GetOpAttr(const std::string &op_name, const std::string &attr_name, std::string &attr_value) const;

  void GetModelAttr(vector<string> &dynamic_output_shape_info) const;

  ///
  /// @ingroup ge
  /// @brief Get AIPP input info
  /// @param [in] index
  /// @param [out] aipp_info
  /// @return execute result
  ///
  Status GetAippInfo(uint32_t index, AippConfigInfo &aipp_info) const;

  Status GetAippType(uint32_t index, InputAippType &type, size_t &aipp_index) const;

  ///
  /// @ingroup ge
  /// @brief Get model_id.
  /// @return model_id
  ///
  uint32_t GetModelId() const { return model_id_; }

  ///
  /// @ingroup ge
  /// @brief get unique identification for op when load two or more models
  /// @param [in] op_desc : current op.
  /// @param [in] string identification: unique identification for current op.
  /// @return None
  ///
  void GetUniqueId(const OpDescPtr &op_desc, string &unique_identification);

  Status ReturnResult(uint32_t data_id, const bool rslt_flg, const bool seq_end_flg, OutputData *output_data);

  Status ReturnNoOutput(uint32_t data_id);

  Status ModelRunStart();

  ///
  /// @ingroup ge
  /// @brief stop run model
  /// @return Status
  ///
  Status ModelRunStop();

  ///
  /// @ingroup ge
  /// @brief model run flag
  /// @return Status
  ///
  bool RunFlag() const { return run_flg_; }

  ///
  /// @ingroup ge
  /// @brief Set Session Id
  /// @return void
  ///
  void SetSessionId(uint64_t session_id) { session_id_ = session_id; }

  ///
  /// @ingroup ge
  /// @brief Get Session Id
  /// @return sessionID
  ///
  uint64_t GetSessionId() const { return session_id_; }

  const struct error_message::Context &GetErrorContext() const { return error_context_; }

  ///
  /// @ingroup ge
  /// @brief SetDeviceId
  /// @return void
  ///
  void SetDeviceId(uint32_t device_id) { device_id_ = device_id; }

  ///
  /// @ingroup ge
  /// @brief Get device Id
  /// @return  device id
  ///
  uint32_t GetDeviceId() const { return device_id_; }

  bool NeedDestroyAicpuKernel() const { return need_destroy_aicpu_kernel_; }

  Status UpdateSessionId(uint64_t session_id);

  const RuntimeParam &GetRuntimeParam() { return runtime_param_; }

  int32_t GetDataInputTid() const { return dataInputTid; }
  void SetDataInputTid(int32_t data_input_tid) { dataInputTid = data_input_tid; }

  void DisableZeroCopy(const void *addr);

  bool GetOpDugReg() const { return is_op_debug_reg_; }

  ///
  /// @ingroup ge
  /// @brief Save outside address of Data or NetOutput used info for ZeroCopy.
  /// @param [in] const OpDescPtr &op_desc: current op desc
  /// @param [in] const vector<void *> &outside_addrs: address of task
  /// @param [in] const void *args_offset: arguments address save the address.
  /// @return None.
  ///
  void SetZeroCopyAddr(const OpDescPtr &op_desc, const vector<void *> &outside_addrs, const void *info, void *args,
                       size_t size, size_t offset);

  void SetDynamicSize(const vector<uint64_t> &batch_num, int32_t dynamic_type);

  bool GetL1FusionEnableOption() { return is_l1_fusion_enable_; }

  void SetProfileTime(ModelProcStage stage, int64_t endTime = 0);

  int64_t GetLoadBeginTime() { return load_begin_time_; }

  int64_t GetLoadEndTime() { return load_end_time_; }

  void SaveSpecifyAttrValues(const OpDescPtr &op_desc);

  Status ReportProfilingData();

  void SaveDumpOpInfo(const RuntimeParam &model_param, const OpDescPtr &op, uint32_t task_id, uint32_t stream_id) {
    exception_dumper_.SaveDumpOpInfo(model_param, op, task_id, stream_id);
  }

  void SaveDumpTask(uint32_t task_id, uint32_t stream_id, const shared_ptr<OpDesc> &op_desc, uintptr_t args) {
    data_dumper_.SaveDumpTask(task_id, stream_id, op_desc, args);
  }

  Status DumpExceptionInfo(const std::vector<rtExceptionInfo> &exception_infos) const {
    return exception_dumper_.DumpExceptionInfo(exception_infos);
  }

  void DumperShrink() {
    data_dumper_.DumpShrink();
  }

  bool OpNeedDump(const string &op_name) {
    return GetDumpProperties().IsLayerNeedDump(dump_model_name_, om_name_, op_name);
  }

  bool ModelNeedDump();

  void SetEndGraphId(uint32_t task_id, uint32_t stream_id);
  DavinciModel &operator=(const DavinciModel &model) = delete;

  DavinciModel(const DavinciModel &model) = delete;

  const map<int64_t, vector<rtStream_t>> &GetHcclFolowStream() {
    return main_follow_stream_mapping_;
  }
  void SaveHcclFollowStream(int64_t main_stream_id, rtStream_t stream);

  void InitRuntimeParams();
  Status InitVariableMem();

  void UpdateMemBase(uint8_t *mem_base) {
    runtime_param_.mem_base = mem_base;
    mem_base_ = mem_base;
  }
  void SetTotalArgsSize(uint32_t args_size) { total_args_size_ += args_size; }
  uint32_t GetTotalArgsSize() { return total_args_size_; }
  void *GetCurrentArgsAddr(uint32_t offset) {
    void *cur_args = static_cast<char *>(args_) + offset;
    return cur_args;
  }
  void SetTotalIOAddrs(const vector<void *> &io_addrs);
  void SetHybridArgsSize(uint32_t args_size) { total_hybrid_args_size_ += args_size; }
  uint32_t GetHybridArgsSize() {
    return total_hybrid_args_size_;
  }
  void *GetCurrentHybridArgsAddr(uint32_t offset) {
    void *cur_args = static_cast<char *>(hybrid_addrs_) + offset;
    return cur_args;
  }
  void SetTotalFixedAddrsSize(string tensor_name, int64_t fix_addr_size);
  int64_t GetFixedAddrsSize(string tensor_name);
  void *GetCurrentFixedAddr(int64_t offset) const {
    void *cur_addr = static_cast<char *>(fixed_addrs_) + offset;
    return cur_addr;
  }

  uint32_t GetFixedAddrOutputIndex(string tensor_name) {
    if (tensor_name_to_peer_output_index_.find(tensor_name) != tensor_name_to_peer_output_index_.end()) {
      return tensor_name_to_peer_output_index_[tensor_name];
    }
    return UINT32_MAX;
  }
  void SetKnownNode(bool known_node) { known_node_ = known_node; }
  bool IsKnownNode() { return known_node_; }
  Status MallocKnownArgs();
  Status CheckCapability(rtFeatureType_t featureType, int32_t featureInfo, bool &is_support) const;
  Status UpdateKnownNodeArgs(const vector<void *> &inputs, const vector<void *> &outputs);
  Status CreateKnownZeroCopyMap(const vector<void *> &inputs, const vector<void *> &outputs);
  Status UpdateKnownZeroCopyAddr(vector<void *> &total_io_addrs, bool update_args = true);

  Status GetOrigInputInfo(uint32_t index, OriginInputInfo &orig_input_info) const;
  Status GetAllAippInputOutputDims(uint32_t index, vector<InputOutputDims> &input_dims,
                                   vector<InputOutputDims> &output_dims) const;

  // om file name
  void SetOmName(const string &om_name) { om_name_ = om_name; }
  void SetDumpModelName(const string &dump_model_name) { dump_model_name_ = dump_model_name; }

  void SetDumpProperties(const DumpProperties &dump_properties) { data_dumper_.SetDumpProperties(dump_properties); }
  const DumpProperties &GetDumpProperties() const { return data_dumper_.GetDumpProperties(); }

  bool GetOpDescInfo(uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info) const {
    return exception_dumper_.GetOpDescInfo(stream_id, task_id, op_desc_info);
  }
  void UpdateOpIOAddrs(uint32_t task_id, uint32_t stream_id, const std::vector<void *> &io_addrs);

  bool GetRunningFlag() const { return running_flg_; }
  void SetRunningFlag(bool flag) { running_flg_ = flag; }
  Status SetRunAsyncListenerCallback(const RunAsyncCallback &callback);

  // for blocking aicpu op
  Status GetEventByStream(const rtStream_t &stream, rtEvent_t &rt_event);
  Status GetEventIdForBlockingAicpuOp(const OpDescPtr &op_desc, rtStream_t stream, uint32_t &event_id);

 private:
  // memory address of weights
  uint8_t *weights_mem_base_;
  uint8_t *var_mem_base_;
  // memory address of model
  uintptr_t fixed_mem_base_;  // Initial of mem_base_, keep forever.
  uint8_t *mem_base_;
  bool is_inner_mem_base_;
  bool is_inner_weight_base_;
  // input data manager
  DataInputer *data_inputer_;
  int64_t load_begin_time_;
  int64_t load_end_time_;
  struct timeInfo time_info_;
  int32_t dataInputTid;

  void *GetRunAddress(void *addr) const;

  ///
  /// @ingroup ge
  /// @brief Copy Check input size and model op size.
  /// @param [in] const int64_t &input_size: input size.
  /// @param [in] const int64_t &op_size: model op size.
  /// @param [in] is_dynamic: dynamic batch input flag.
  /// @return true if success
  ///
  bool CheckUserAndModelSize(const int64_t &size, const int64_t &op_size, bool is_input, bool is_dynamic);

  ///
  /// @ingroup ge
  /// @brief Set copy only for No task feed NetOutput address.
  /// @return None.
  ///
  void SetCopyOnlyOutput();

  ///
  /// @ingroup ge
  /// @brief Copy Input/Output to model for direct use.
  /// @param [in] const InputData &input_data: user input data info.
  /// @param [in/out] OutputData &output_data: user output data info.
  /// @param [in] bool is_dynamic: whether is dynamic input, true: is dynamic input; false: not is dynamic input
  /// @return SUCCESS handle successfully / others handle failed
  ///
  Status CopyModelData(const InputData &input_data, OutputData &output_data, bool is_dynamic);

  ///
  /// @ingroup ge
  /// @brief Copy Data addr to model for direct use.
  /// @param [in] data_info: model memory addr/size map { data_index, { tensor_size, tensor_addr } }.
  /// @param [in] is_input: input data or output data
  /// @param [in] blobs: user input/output data list.
  /// @param [in] is_dynamic: whether is dynamic input, true: is dynamic input; false: not is dynamic input
  /// @param [in] batch_label: batch label for multi-batch scenes
  /// @return SUCCESS handle successfully / others handle failed
  ///
  Status UpdateIoTaskArgs(const map<uint32_t, ZeroCopyOffset> &data_info, bool is_input,
                          const vector<DataBuffer> &blobs, bool is_dynamic, const string &batch_label);

  Status CopyInputData(const InputData &input_data);

  Status CopyOutputData(uint32_t data_id, OutputData &output_data, rtMemcpyKind_t kind);

  Status SyncVarData();

  Status InitWeightMem(void *dev_ptr, void *weight_ptr, size_t weight_size);
  Status InitFeatureMapAndP2PMem(void *dev_ptr, size_t mem_size);

  void CreateInputDimsInfo(const OpDescPtr &op_desc, Format format, ShapeDescription &shape1, ShapeDescription &shape2);

  void SetInputDimsInfo(const vector<int64_t> &input_dims, Format &format, ShapeDescription &shape_info);

  Status GetInputDescInfo(vector<InputOutputDescInfo> &input_desc, vector<uint32_t> &input_formats, bool by_dims) const;
  Status GetOutputDescInfo(vector<InputOutputDescInfo> &output_desc, vector<uint32_t> &output_formats) const;

  Status InitTaskInfo(domi::ModelTaskDef &modelTaskInfo);

  void UnbindHcomStream();

  Status DistributeTask();

  void SaveProfilingTaskDescInfo(const OpDescPtr &op, const TaskInfoPtr &task,
                                 const domi::TaskDef &task_def, size_t task_index);

  uint8_t *MallocFeatureMapMem(size_t data_size);

  uint8_t *MallocWeightsMem(size_t weights_size);

  Status MallocExMem();

  void FreeFeatureMapMem();

  void FreeWeightsMem();

  void FreeExMem();

  void ReleaseTask();

  void ClearTaskAddrs();

  void UnbindTaskSinkStream();

  bool IsAicpuKernelConnectSpecifiedLayer();

  ///
  /// @ingroup ge
  /// @brief Reduce memory usage after task sink.
  /// @return: void
  ///
  void Shrink();

  ///
  /// @ingroup ge
  /// @brief Travel all nodes and do some init.
  /// @param [in] compute_graph: ComputeGraph to load.
  /// @return Status
  ///
  Status InitNodes(const ComputeGraphPtr &compute_graph);

  ///
  /// @ingroup ge
  /// @brief Data Op Initialize.
  /// @param [in] ComputeGraphPtr: root graph of the model.
  /// @param [in] NodePtr: Data Op.
  /// @param [in/out] data_op_index: index of courrent count.
  /// @param [in/out] data_by_index: Data ordered by index.
  /// @return Status
  ///
  Status InitDataOp(const ComputeGraphPtr &graph, const NodePtr &node, uint32_t &data_op_index,
                    map<uint32_t, OpDescPtr> &data_by_index, set<const void *> &input_outside_addrs);

  ///
  /// @ingroup ge
  /// @brief Sort Data op list by index.
  /// @param [in] data_by_index: map of Data Op.
  /// @param [in] output_op_list: list of NetOutput op.
  /// @return Status
  ///
  Status GenInputOutputInfo(const map<uint32_t, OpDescPtr> &data_by_index, const vector<OpDescPtr> &output_op_list);

  ///
  /// @ingroup ge
  /// @brief NetOutput Op Initialize.
  /// @param [in] ComputeGraphPtr: root graph of the model.
  /// @param [in] NodePtr: NetOutput Op.
  /// @param [in/out] vector<OpDescPtr>: All NetOutput node in model.
  /// @return Status
  ///
  Status InitNetOutput(const ComputeGraphPtr &graph, const NodePtr &node, vector<OpDescPtr> &output_op_list,
                       set<const void *> &output_outside_addrs);

  ///
  /// @ingroup ge
  /// @brief Constant Op Init.
  /// @return Status
  ///
  Status InitConstant(const OpDescPtr &op_desc);

  Status InitVariable(const OpDescPtr &op_desc, map<string, OpDescPtr> &variable_by_name);

  /// @ingroup ge
  /// @brief LabelSet Op Initialize.
  /// @param [in] op_desc: LabelSet Op descriptor.
  /// @return Status
  Status InitLabelSet(const OpDescPtr &op_desc);

  Status InitStreamSwitch(const OpDescPtr &op_desc);

  Status InitStreamActive(const OpDescPtr &op_desc);

  Status InitStreamSwitchN(const OpDescPtr &op_desc);

  ///
  /// @ingroup ge
  /// @brief Case Op Init.
  /// @return Status
  ///
  Status InitCase(const OpDescPtr &op_desc);

  Status SetDynamicBatchInfo(const OpDescPtr &op_desc, uint32_t batch_num);

  ///
  /// @ingroup ge
  /// @brief TVM Op Init.
  /// @return Status
  ///
  Status InitTbeHandle(const OpDescPtr &op_desc);
  Status InitTbeHandleWithFfts(const OpDescPtr &op_desc);
  Status FunctionRegister(const OpDescPtr &op_desc, string &bin_file, OpKernelBinPtr &tbe_kernel, bool is_ffts,
                          size_t thread_index = 0);
  Status InitBinaryMagic(const OpDescPtr &op_desc, bool is_ffts, size_t thread_index, rtDevBinary_t &binary);
  Status InitMetaData(const OpDescPtr &op_desc, bool is_ffts, size_t thread_index, void *bin_handle);
  Status InitKernelName(const OpDescPtr &op_desc, bool is_ffts, size_t thread_index, string &kernel_name);

  void StoreTbeHandle(const string &handle_key);
  void CleanTbeHandle();

  ///
  /// @ingroup ge
  /// @brief Make active stream list and bind to model.
  /// @return: 0 for success / others for fail
  ///
  Status BindModelStream();

  ///
  /// @ingroup ge
  /// @brief Init model stream for NN model.
  /// @return Status
  ///
  Status InitModelStream(rtStream_t stream);

  ///
  /// @ingroup ge
  /// @brief ACL, Load task list with queue entrance.
  /// @return: 0 for success / others for fail
  ///
  Status LoadWithQueue();

  ///
  /// @ingroup ge
  /// @brief ACL, Bind Data Op addr to input queue.
  /// @return: 0 for success / others for fail
  ///
  Status BindInputQueue();

  Status CpuTaskModelZeroCopy(vector<uintptr_t> &mbuf_list, const map<uint32_t, ZeroCopyOffset> &outside_addrs);

  ///
  /// @ingroup ge
  /// @brief ACL, Bind NetOutput Op addr to output queue.
  /// @return: 0 for success / others for fail
  ///
  Status BindOutputQueue();
  Status CpuModelPrepareOutput(uintptr_t addr, uint32_t size);

  ///
  /// @ingroup ge
  /// @brief definiteness queue schedule, bind input queue to task.
  /// @param [in] queue_id: input queue id from user.
  /// @param [in] addr: Data Op output tensor address.
  /// @param [in] size: Data Op output tensor size.
  /// @return: 0 for success / others for fail
  ///
  Status CpuModelDequeue(uint32_t queue_id);

  ///
  /// @ingroup ge
  /// @brief definiteness queue schedule, bind output queue to task.
  /// @param [in] queue_id: output queue id from user.
  /// @param [in] addr: NetOutput Op input tensor address.
  /// @param [in] size: NetOutput Op input tensor size.
  /// @return: 0 for success / others for fail
  ///
  Status CpuModelEnqueue(uint32_t queue_id, uintptr_t addr, uint32_t size);

  ///
  /// @ingroup ge
  /// @brief definiteness queue schedule, active original model stream.
  /// @return: 0 for success / others for fail
  ///
  Status CpuActiveStream();

  ///
  /// @ingroup ge
  /// @brief definiteness queue schedule, wait for end graph.
  /// @return: 0 for success / others for fail
  ///
  Status CpuWaitEndGraph();

  Status BindEnqueue();
  Status CpuModelEnqueue(uint32_t queue_id, uintptr_t out_mbuf);
  ///
  /// @ingroup ge
  /// @brief definiteness queue schedule, repeat run model.
  /// @return: 0 for success / others for fail
  ///
  Status CpuModelRepeat();

  Status InitEntryTask();
  Status AddHeadStream();

  ///
  /// @ingroup ge
  /// @brief set ts device.
  /// @return: 0 for success / others for fail
  ///
  Status SetTSDevice();

  Status OpDebugRegister();

  void OpDebugUnRegister();

  void CheckHasHcomOp(const ComputeGraphPtr &graph);

  Status DoTaskSink();

  void CreateOutput(uint32_t index, const OpDescPtr &op_desc, InputOutputDescInfo &output, uint32_t &format_result);

  Status TransAllVarData(ComputeGraphPtr &graph, uint32_t graph_id);

  void SetDataDumperArgs(const ComputeGraphPtr &graph, const map<string, OpDescPtr> &variable_by_name);

  Status InitL1DataDumperArgs();

  Status InitModelProfile();
  Status SinkModelProfile();

  Status SinkTimeProfile(const InputData &current_data);

  Status InitOutputTensorInfo(const OpDescPtr &op_desc);
  Status GenOutputTensorInfo(OutputData *output_data, vector<ge::Tensor> &outputs);

  Status InitInputDescInfo(const OpDescPtr &op_desc);
  Status InitOutputDescInfo(const OpDescPtr &op_desc, const vector<string> &out_node_name);

  Status InitOrigInputInfo(uint32_t index, const OpDescPtr &op_desc);
  Status InitAippInfo(uint32_t index, const OpDescPtr &op_desc);
  Status InitAippType(uint32_t index, const OpDescPtr &op_desc, const map<uint32_t, OpDescPtr> &data_list);
  Status InitAippInputOutputDims(uint32_t index, const OpDescPtr &op_desc);

  void ParseAIPPInfo(string in_out_info, InputOutputDims &dims_info);
  void SetLabelForDynamic(const NodePtr &node);

  void ParseDynamicOutShape(const vector<string> &str_info, vector<vector<int64_t>> &vec_info);
  bool IsGetNextSinkDynamic(const OpDescPtr &op_desc);

  Status InitRealSizeAndShapeInfo(const ComputeGraphPtr &compute_graph, const NodePtr &node);
  void GetAllGearsInfo(const NodePtr &node);
  Status GetGetDynamicDimsNodeInfo(const NodePtr &node);
  Status GetGearAndRealOutSizeInfo(const ComputeGraphPtr &graph, const NodePtr &node);
  Status GetRealOutputSizeOfCase(const ComputeGraphPtr &graph, size_t input_index, const NodePtr &case_node);
  Status GetGearAndRealOutShapeInfo(const ComputeGraphPtr &graph, const NodePtr &node);

  bool is_weight_mem_has_inited_;
  bool is_feature_map_mem_has_inited_;

  uint32_t model_id_;
  uint32_t runtime_model_id_;
  uint32_t sub_model_id_ = 0;
  string name_;

  // used for inference data dump
  string om_name_;
  string dump_model_name_;

  uint32_t version_;
  GeModelPtr ge_model_;  // release after DavinciModel::Init

  bool need_destroy_aicpu_kernel_{false};

  map<uint32_t, OpDescPtr> op_list_;  // release after DavinciModel::Init

  map<string, GeTensorDesc> broadcast_variable_;
  void *global_step_addr_{nullptr};
  uint64_t global_step_size_{0};

  map<uint32_t, ZeroCopyOffset> input_data_info_;
  map<uint32_t, ZeroCopyOffset> output_data_info_;

  set<const void *> real_virtual_addrs_;

  // output op: save cce op actual needed memory size
  vector<int64_t> output_memory_size_list_;

  thread thread_id_;

  shared_ptr<ModelListener> listener_;

  bool run_flg_;
  // check whether model is running with data
  bool running_flg_ = false;

  mutex mux_run_flg_;

  int32_t priority_;

  vector<rtStream_t> stream_list_;

  mutex all_hccl_stream_list_mutex_;
  vector<rtStream_t> all_hccl_stream_list_;

  // for reuse hccl_follow_stream
  mutex capacity_of_stream_mutex_;
  map<int64_t, vector<rtStream_t>> main_follow_stream_mapping_;

  vector<rtEvent_t> event_list_;

  vector<rtLabel_t> label_list_;
  set<uint32_t> label_id_indication_;

  mutex label_args_mutex_;
  map<uint32_t, pair<void *, uint32_t>> label_goto_args_;

  mutex outside_addrs_mutex_;
  vector<ZeroCopyTask> zero_copy_tasks_;  // Task used Data or NetOutput addr.
  set<const void *> copy_only_addrs_;     // Address need copy to original place.

  vector<TaskInfoPtr> task_list_;
  // rt_moodel_handle
  rtModel_t rt_model_handle_;

  rtStream_t rt_model_stream_;

  bool is_inner_model_stream_;

  bool is_async_mode_;  // For NN execute, Async mode use rtMemcpyAsync on rt_model_stream_.
  ExecuteMode last_execute_mode_;

  bool is_stream_list_bind_{false};
  bool is_pure_head_stream_{false};
  rtStream_t rt_head_stream_{nullptr};
  rtStream_t rt_entry_stream_{nullptr};
  rtAicpuDeployType_t deploy_type_{AICPU_DEPLOY_RESERVED};

  // ACL queue schedule, save queue ids for Init.
  vector<TaskInfoPtr> cpu_task_list_;
  vector<uint32_t> input_queue_ids_;    // input queue ids created by caller.
  vector<uint32_t> output_queue_ids_;   // output queue ids created by caller.
  vector<uintptr_t> input_mbuf_list_;   // input mbuf created by dequeue task.
  vector<uintptr_t> output_mbuf_list_;  // output mbuf created by dequeue task.

  uint64_t session_id_;
  struct error_message::Context error_context_;

  uint32_t device_id_;

  mutex flowctrl_op_index_internal_map_mutex_;
  map<uint32_t, uint32_t> flowctrl_op_index_internal_map_;

  vector<rtStream_t> active_stream_list_;
  set<uint32_t> active_stream_indication_;

  set<uint32_t> hcom_streams_;
  RuntimeParam runtime_param_;

  static mutex tvm_bin_mutex_;
  set<string> tvm_bin_kernel_;

  map<string, uint32_t> used_tbe_handle_map_;

  // for profiling task and graph info
  vector<TaskDescInfo> task_desc_info_;

  std::map<std::string, std::pair<uint32_t, uint32_t>> profiler_report_op_info_;

  int64_t maxDumpOpNum_;
  // for data dump
  DataDumper data_dumper_;
  ExceptionDumper exception_dumper_;
  OpdebugRegister opdebug_register_;
  uint64_t iterator_count_;
  bool is_l1_fusion_enable_;
  map<OpDescPtr, void *> saved_task_addrs_;  // release after DavinciModel::Init
  void *l1_fusion_addr_ = nullptr;

  bool known_node_ = false;
  uint32_t total_args_size_ = 0;
  void *args_ = nullptr;
  void *args_host_ = nullptr;
  void *fixed_addrs_ = nullptr;
  void *hybrid_addrs_ = nullptr;
  uint32_t total_hybrid_args_size_ = 0;
  int64_t total_fixed_addr_size_ = 0;
  map<const void *, void *> known_input_data_info_;
  map<const void *, void *> known_output_data_info_;
  vector<void *> total_io_addrs_;

  vector<vector<int64_t>> batch_info_;
  vector<vector<int64_t>> combined_batch_info_;
  vector<string> user_designate_shape_order_;
  int32_t dynamic_type_ = 0;
  bool is_dynamic_ = false;

  vector<uint64_t> batch_size_;
  // key: input tensor name, generally rts op;
  // value: the fixed addr of input anchor, same as the peer output anchor addr of the peer op
  map<string, int64_t> tensor_name_to_fixed_addr_size_;

  // key: input tensor name, generally rts op; value: the peer output anchor of the peer op
  map<string, int64_t> tensor_name_to_peer_output_index_;
  // if model is first execute
  bool is_first_execute_;
  // for op debug
  mutex debug_reg_mutex_;
  bool is_op_debug_reg_ = false;
  bool is_online_infer_dynamic_ = false;
  bool is_getnext_sink_dynamic_ = false;
  vector<int32_t> cur_dynamic_dims_;
  void *netoutput_last_input_addr_ = nullptr;
  int64_t netoutput_last_input_size_ = 0;
  size_t shape_of_cur_dynamic_dims_ = 0;
  // key: input_index: input is merge node; value: each gear info and each output size
  map<size_t, map<vector<int32_t>, int64_t>> merge_nodes_gear_and_real_out_size_info_;
  // key: input_index: input is merge node; value: each gear info and each output shape
  map<size_t, map<vector<int32_t>, vector<int64_t>>> merge_nodes_gear_and_real_out_shape_info_;
  vector<vector<int32_t>> all_gears_info_;

  multimap<uint32_t, uint32_t> op_id_map_;
  vector<ProfileInfo> profile_list_;

  // For super kernel.
  SuperKernelTaskInfo skt_info_;

  bool has_output_node_ = false;
  bool is_dynamic_aipp_ = false;
  vector<string> dynamic_output_shape_info_;

  vector<vector<void *>> input_addrs_list_;
  vector<vector<void *>> output_addrs_list_;

  vector<int64_t> output_buffer_size_;
  vector<GeShape> output_shape_info_;

  map<uint32_t, OriginInputInfo> orig_input_info_;
  map<uint32_t, AippConfigInfo> aipp_info_list_;
  map<uint32_t, pair<InputAippType, size_t>> aipp_type_list_;
  map<uint32_t, pair<vector<InputOutputDims>, vector<InputOutputDims>>> aipp_dims_info_;

  vector<InputOutputDescInfo> input_descs_;
  vector<InputOutputDescInfo> input_descs_dims_;
  vector<uint32_t> input_formats_;
  vector<InputOutputDescInfo> output_descs_;
  vector<uint32_t> output_formats_;

  // op name to attrs mapping
  std::map<std::string, std::map<std::string, std::vector<std::string>>> op_name_to_attrs_;

  std::map<rtStream_t, rtEvent_t> stream_2_event_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_H_
