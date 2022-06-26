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
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include <locale>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <set>
#include "include/common/utils/signal_util.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/device/distribute/ascend_collective.h"
#include "utils/ms_context.h"
#include "runtime/device/context_extends.h"
#include "include/common/utils/mpi/mpi_config.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/rt.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/hal/device/ge_runtime/model_runner.h"
#include "plugin/device/ascend/hal/device/tasksink/task_generator.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_build_client.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#endif
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "debug/data_dump/e2e_dump.h"
#endif
#include "toolchain/adx_datadump_server.h"
#include "utils/trace_base.h"
#include "graphengine/inc/external/acl/error_codes/rt_error_codes.h"
#include "common/util/error_manager/error_manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/utils/runtime_error_codes.h"
#ifdef MEM_REUSE_DEBUG
#include "common/mem_reuse/mem_reuse_checker.h"
#include "include/common/debug/env_config_parser.h"
#endif
#include "plugin/device/ascend/hal/device/executor/hccl_dynamic_kernel.h"
#include "include/common/utils/config_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#ifdef ENABLE_TDTQUE
#include "minddata/dataset/engine/tdt/tdt_handle.h"
using luojianet_ms::dataset::TdtHandle;
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#endif

#include "backend/common/session/pynative_task_manager.h"
#include "profiler/device/profiling.h"

#ifndef ENABLE_SECURITY
using luojianet_ms::device::ascend::ProfilingManager;
using luojianet_ms::device::ascend::ProfilingUtils;
#endif
using luojianet_ms::device::ascend::tasksink::TaskGenerator;
using luojianet_ms::ge::model_runner::ModelRunner;
using luojianet_ms::kernel::tbe::TbeUtils;
using std::vector;

constexpr uint32_t kTupleTaskId = 0;
constexpr uint32_t kTupleStreamId = 1;
constexpr uint32_t kTupleArgs = 2;
constexpr uint32_t kProfilingMaxTaskIdInStream = 65531;
constexpr auto kModuleName = "LuoJiaNET";
constexpr size_t kPathMax = 4096;

namespace luojianet_ms::device::ascend {
static thread_local rtContext_t thread_local_rt_context{nullptr};
constexpr auto kUnknowErrorString = "Unknown error occurred";
namespace {
std::string GetRankIdStr() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
    MS_LOG(INFO) << "Get hccl rankid from mpi";
    auto rank = HcclCollectiveGroup::instance().GetRankId();
    return std::to_string(rank);
  }
  auto rank_id_str = common::GetEnv("RANK_ID");
  if (rank_id_str.empty()) {
    MS_LOG(EXCEPTION) << "Invalid environment variable 'RANK_ID', it should not be empty.";
  }
  return rank_id_str;
}

void IntHandler(int, siginfo_t *, void *) {
  luojianet_ms::kernel::AscendKernelBuildClient::Instance().Close();
  int this_pid = getpid();
  MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
  (void)kill(this_pid, SIGTERM);
}

void AscendEnableDynamicRuntimeCache(const session::KernelGraph *graph) {
  const auto &node_list = graph->TopoSort(graph->get_return());
  for (auto &node : node_list) {
    auto kernel_info = node->kernel_info();
    if (!kernel_info) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(kernel_info);
    auto runtime_cache = kernel_info->runtime_cache();
    runtime_cache.runtime_cache().set_valid();
  }
}
}  // namespace

std::vector<rtExceptionInfo> AscendKernelRuntime::task_fail_infoes_ = {};
const session::KernelGraph *current_graph_ = nullptr;
std::map<std::string, uint32_t> AscendKernelRuntime::overflow_tasks_;
AscendKernelRuntime::~AscendKernelRuntime() {
  graph_model_map_.clear();
  current_graph_ = nullptr;
  rt_context_ = nullptr;
}

void AscendKernelRuntime::SetContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  if (thread_local_rt_context == rt_context_) {
    return;
  }
  auto ret = rtCtxSetCurrent(rt_context_);
  thread_local_rt_context = rt_context_;
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtCtxSetCurrent, ret[" << ret << "]";
  }
}

void AscendKernelRuntime::SetCurrentContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  auto ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtCtxSetCurrent, ret[" << ret << "]";
  }
}

void AscendKernelRuntime::ClearGraphModelMap() {
  SetCurrentContext();
#ifndef ENABLE_SECURITY
  for (auto &iter : graph_data_dumper_) {
    MS_LOG(INFO) << "[DataDump] Unload data dumper:" << iter.first;
    auto &data_dumper = iter.second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->UnloadDumpInfo();
    data_dumper->OpDebugUnregister();
  }
  graph_data_dumper_.clear();
  // tell users which dump kernel name not used
  DumpJsonParser::GetInstance().PrintUnusedKernel();
#endif

  graph_dynamic_kernel_map_.clear();
  graph_kernel_events_map_.clear();
  for (auto &iter : graph_model_map_) {
    MS_LOG(INFO) << "Ge UnloadModel " << iter.first;
    ModelRunner::Instance().UnloadModel(iter.first);
  }
}

void AscendKernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
  SetCurrentContext();
  auto mem_scheduler = mem_scheduler_manager_.GetMemScheduler(graph_id);
  if (mem_scheduler != nullptr) {
    mem_scheduler->Clear();
  }
  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " data dumper";
#ifndef ENABLE_SECURITY
  if (auto dumper_iter = graph_data_dumper_.find(graph_id); dumper_iter != graph_data_dumper_.end()) {
    MS_LOG(DEBUG) << "Unload dump info " << graph_id;
    auto &data_dumper = dumper_iter->second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->UnloadDumpInfo();
    data_dumper->OpDebugUnregister();
    graph_data_dumper_.erase(dumper_iter);
  } else {
    MS_LOG(DEBUG) << "GraphId:" << graph_id << " not found";
  }
#endif

  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " dynamic kernels";
  if (auto dynamic_kernel_iter = graph_dynamic_kernel_map_.find(graph_id);
      dynamic_kernel_iter != graph_dynamic_kernel_map_.end()) {
    MS_LOG(DEBUG) << "Start Clear graph:" << graph_id << " dynamic kernel";
    graph_dynamic_kernel_map_.erase(dynamic_kernel_iter);
  }
  auto events_iter = graph_kernel_events_map_.find(graph_id);
  if (events_iter != graph_kernel_events_map_.end()) {
    graph_kernel_events_map_.erase(events_iter);
  }
  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " runtime resource";
  if (auto model_iter = graph_model_map_.find(graph_id); model_iter != graph_model_map_.end()) {
    MS_LOG(DEBUG) << "Ge UnloadModel " << graph_id;
    ModelRunner::Instance().UnloadModel(graph_id);
    graph_model_map_.erase(model_iter);
  } else {
    MS_LOG(DEBUG) << "GraphId:" << graph_id << " not found";
  }
}

void *AscendKernelRuntime::GetModelStream(uint32_t graph_id) const {
  return ModelRunner::Instance().GetModelStream(graph_id);
}

void AscendKernelRuntime::ClearGlobalIdleMem() {
  if (mem_manager_ != nullptr) {
    mem_manager_->ClearGlobalIdleMem();
  }
}

bool AscendKernelRuntime::NeedDestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    MS_LOG(INFO) << "Hccl is not enabled";
    return false;
  }
  // Note: make sure hcom_connectivity_detection api never be used.
  return true;
}

#ifndef ENABLE_SECURITY
void AsyncDataDumpUninit() {
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
#if ENABLE_D
    // When it is A+M dump mode, wait until file save is finished.
    if (DumpJsonParser::GetInstance().FileFormatIsNpy()) {
      Debugger::GetInstance()->WaitForWriteFileFinished();
    }
#endif
    if (AdxDataDumpServerUnInit() != 0) {
      MS_LOG(ERROR) << "Adx data dump server uninit failed";
    }
  }
}
#endif

void AscendKernelRuntime::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Ascend finalize start";
#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->debugger_enabled()) {
    debugger_->SetTrainingDone(true);
    bool ret = debugger_->SendMetadata(false);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to SendMetadata when finalize";
    }
  }
#endif
  if (!initialized_) {
    return;
  }
  SetCurrentContext();

  // release ge runtime
  ClearGraphModelMap();

#ifndef ENABLE_SECURITY
  AsyncDataDumpUninit();
#endif

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  // DestroyHccl must be called before FreeDeviceMemory
  (void)DestroyHccl();
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
  }
  luojianet_ms::kernel::AicpuOpKernelLoad::GetInstance().FreeDeviceMemory();

  auto rt_ret = rtRegTaskFailCallbackByModule(kModuleName, nullptr);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
  }

  (void)ResetDevice(device_id);
  current_graph_ = nullptr;
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      !context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
    HcclCollectiveGroup::instance().FinalizeCollective();
  }
  initialized_ = false;
  MS_LOG(INFO) << "Ascend finalize end";
}

#ifndef ENABLE_SECURITY
void AscendKernelRuntime::PreInit() {
  const auto error_manager_ret = ErrorManager::GetInstance().Init();
  if (error_manager_ret != 0) {
    MS_LOG(WARNING) << "Init ErrorManager failed.";
  }
}
#endif

bool AscendKernelRuntime::Init() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
#ifndef ENABLE_SECURITY
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  auto profiling_flag = profiler_manager->GetProfilingEnableFlag();
  if (execution_mode == kPynativeMode && profiling_flag) {
    pynative_mode_profiling_flag_ = true;
  }
#endif
  if (initialized_) {
    SetCurrentContext();
    return true;
  }
  const auto error_manager_ret = ErrorManager::GetInstance().Init();
  if (error_manager_ret != 0) {
    MS_LOG(WARNING) << "Init ErrorManager failed.";
  }
  try {
    // Start up profiling before rtSetDevice
    bool ret = InitDevice();
    if (!ret) {
      return ret;
    }
#ifdef ENABLE_DEBUGGER
    SetDebugger();
#endif
    mem_manager_ = std::make_shared<AscendMemoryManager>();
    MS_EXCEPTION_IF_NULL(mem_manager_);
    mem_manager_->Initialize();

    // Set callback func when exception error
    auto rt_ret = rtRegTaskFailCallbackByModule(kModuleName, TaskFailCallback);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
    }
  } catch (const std::exception &e) {
    const string &error_message = ErrorManager::GetInstance().GetErrorMessage();
    if (!error_message.empty() && error_message.find(kUnknowErrorString) == string::npos) {
      MS_LOG(EXCEPTION) << "Ascend error occurred, error message: " << error_message
                        << "\nFirst error scene API: " << e.what();
    }
    throw;
  }

  initialized_ = true;
  return true;
}

bool AscendKernelRuntime::LoadData(const session::KernelGraph &) {
#ifdef ENABLE_DEBUGGER
  MS_LOG(INFO) << "Start load step";
  MS_EXCEPTION_IF_NULL(debugger_);
  for (const auto &graph_ptr : debugger_->GetGraphPtrList()) {
    debugger_->SetGraphPtr(graph_ptr);
    // load output
    debugger_->LoadGraphOutputs();
    // load parameters
    debugger_->LoadParametersAndConst();
  }
#endif
  return true;
}

bool AscendKernelRuntime::KernelMemNotReuse(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  bool need_dump = false;
#ifndef ENABLE_SECURITY
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.e2e_dump_enabled() && dump_json_parser.dump_mode() == 1) {
    auto op_name = node->fullname_with_scope();
    if (dump_json_parser.NeedDump(op_name)) {
      need_dump = true;
    }
  }
#endif
  return need_dump;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto ascend_device_address_ptr =
    std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id, kAscendDevice, device_id);
  ascend_device_address_ptr->set_is_ptr_persisted(true);
  return ascend_device_address_ptr;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id, const KernelWithIndex &node_index) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto ascend_device_address_ptr = std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id,
                                                                         node_index, kAscendDevice, device_id);
  ascend_device_address_ptr->set_is_ptr_persisted(true);
  return ascend_device_address_ptr;
}

bool AscendKernelRuntime::Load(const session::KernelGraph &graph, bool is_task_sink) {
  if (!is_task_sink) {
    MS_LOG(INFO) << "Graph mode with not task sink";
    GenKernelEvents(graph);
    return true;
  }

  if (!GenTask(graph)) {
    return false;
  }
  if (!LoadTask(graph)) {
    return false;
  }
  if (!luojianet_ms::kernel::AicpuOpKernelLoad::GetInstance().LaunchAicpuKernelSo()) {
    return false;
  }
  return true;
}

bool AscendKernelRuntime::GenDynamicKernel(const session::KernelGraph &graph) {
  MS_LOG(INFO) << "GenDynamicKernel start";
  auto cnode_list = graph.execution_order();
  std::vector<DynamicKernelPtr> dynamic_kernels;
  for (const auto &cnode : cnode_list) {
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "Generate node:" << cnode->fullname_with_scope() << " dynamic kernel";
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto dynamic_kernel = kernel_mod->GenDynamicKernel(cnode, stream_);
    if (dynamic_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with the operator [" << common::AnfAlgo::GetCNodeName(cnode)
                        << "].";
    }
    dynamic_kernel->Initialize();
    dynamic_kernels.emplace_back(dynamic_kernel);
  }
  graph_dynamic_kernel_map_[graph.graph_id()] = std::move(dynamic_kernels);
  MS_LOG(INFO) << "GenDynamicKernel end";
  return true;
}

bool AscendKernelRuntime::GenTask(const session::KernelGraph &graph) {
  SetCurrentContext();
  if (graph.is_dynamic_shape()) {
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE && (ConfigManager::GetInstance().iter_num() > 1)) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with dataset_sink_mode.";
    }
#ifndef ENABLE_SECURITY
    if (DumpJsonParser::GetInstance().async_dump_enabled()) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with Asynchronous Dump. Please use Synchronous Dump.";
    }
#endif
    MS_LOG(INFO) << "Dynamic Shape Graph Generate Dynamic kernel";
    return GenDynamicKernel(graph);
  }
  MS_LOG(INFO) << "GenTask start. GraphId:" << graph.graph_id();
#ifndef ENABLE_SECURITY
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    // Update needed dump kernels for old runtime.
    DumpJsonParser::GetInstance().UpdateNeedDumpKernels(graph);
  }
#endif
#ifdef MEM_REUSE_DEBUG
  if (!EnvConfigParser::GetInstance().GetSysMemreuse()) {
    // Get normal graph ir for memreuse
    luojianet_ms::memreuse::MemReuseChecker::GetInstance().CheckNormalIR(&graph);
  }
#endif
  vector<std::shared_ptr<TaskInfo>> task_info_list;
  auto anf_node_list = graph.execution_order();
  auto task_generator = TaskGenerator();
  if (!task_generator.GenTasks(anf_node_list, &task_info_list, graph.graph_id())) {
    return false;
  }
  // Store the task_info_list
  auto insert_ret = task_map_.insert(std::make_pair(graph.graph_id(), task_info_list));
  if (!insert_ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  // Graph may have no compute node, such TensorAddGrad.
  if (task_info_list.empty()) {
    MS_LOG(INFO) << "Graph " << graph.graph_id() << " have no compute node";
    return true;
  }
  AscendStreamAssign &assign_instance = AscendStreamAssign::GetInstance();
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  // the streams' flag not HEAD_STREAM
  std::vector<uint32_t> wait_active_stream_list;
  assign_instance.GetWaitStreams(&wait_active_stream_list);
  std::vector<uint32_t> force_copy_stream_list;
  assign_instance.GetHcomStreams(&force_copy_stream_list);
  MS_LOG(INFO) << "Call DavinciModel total stream num:" << resource_manager.cur_stream_num()
               << ", total event num:" << resource_manager.cur_event_num() << ", total label num:" << graph.label_num()
               << ", wait_active_stream_list size:" << wait_active_stream_list.size()
               << ", force_copy_stream_list size:" << force_copy_stream_list.size();
  auto model = std::make_shared<ge::model_runner::DavinciModel>(
    task_info_list, wait_active_stream_list, force_copy_stream_list, 0, 0, 0, 0, 0, 0,
    resource_manager.cur_stream_num(), graph.label_num(), resource_manager.cur_event_num(), 0);
  auto ret = graph_model_map_.insert(std::make_pair(graph.graph_id(), model));
  if (!ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  MS_LOG(INFO) << "TaskGenerator GetTaskInfo end...";
  return true;
}

bool AscendKernelRuntime::LoadTask(const session::KernelGraph &graph) {
  SetCurrentContext();
  if (graph.is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic Shape Graph Skip Load Task Step";
    return true;
  }

  MS_LOG(INFO) << "LoadTask start. GraphId:" << graph.graph_id();
  if (GraphWithEmptyTaskList(graph)) {
    MS_LOG(INFO) << "LoadTask end, task list is empty";
    return true;
  }

  auto model_iter = graph_model_map_.find(graph.graph_id());
  if (model_iter == graph_model_map_.end()) {
    MS_LOG(ERROR) << "GraphId:" << graph.graph_id() << " Invalid! Graph LoadTask without GenTask.";
    return false;
  }

  MS_LOG(INFO) << "LoadDavinciModel mode_id:" << model_iter->first;
  ModelRunner::Instance().LoadDavinciModel(device_id_, 0, model_iter->first, model_iter->second);

#ifndef ENABLE_SECURITY
  std::function<void *()> model_handle =
    std::bind(&ModelRunner::GetModelHandle, &ModelRunner::Instance(), model_iter->first);
  DistributeDebugTask(graph, NOT_NULL(model_handle));
#endif

  try {
    ModelRunner::Instance().DistributeTask(model_iter->first);
  } catch (const std::exception &e) {
#ifdef ENABLE_DUMP_IR
    luojianet_ms::RDR::TriggerAll();
#endif
    MS_LOG(EXCEPTION) << "Distribute Task Failed, \nerror msg: " << e.what();
  }

#ifndef ENABLE_SECURITY
  if (ProfilingManager::GetInstance().IsProfilingInitialized()) {
    auto task_ids = ModelRunner::Instance().GetTaskIdList(model_iter->first);
    auto stream_ids = ModelRunner::Instance().GetStreamIdList(model_iter->first);
    // Report data directly if profiling is start
    if (ProfilingUtils::ValidComputeGraph(graph)) {
      if (ProfilingManager::GetInstance().IsProfilingStart()) {
        ProfilingUtils::ReportProfilingData(task_ids, stream_ids, graph.graph_id());
      } else {
        // Cache data and save when profiling is start
        ProfilingUtils::SetReportProfilingData(task_ids, stream_ids, graph.graph_id());
      }
    }
  }
  LaunchDataDump(graph.graph_id());
#endif

  ModelRunner::Instance().LoadModelComplete(model_iter->first);
  return true;
}

#ifndef ENABLE_SECURITY
void AscendKernelRuntime::DistributeDebugTask(const session::KernelGraph &graph,
                                              const NotNull<std::function<void *()>> &model_handle) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    return;
  }
  MS_LOG(INFO) << "Start Distribute Debug Task";
  auto data_dumper = std::make_shared<DataDumper>(&graph, model_handle);
  MS_EXCEPTION_IF_NULL(data_dumper);
  auto ret = graph_data_dumper_.try_emplace(graph.graph_id(), data_dumper);
  data_dumper->OpDebugRegister();
  if (!ret.second) {
    MS_LOG(WARNING) << "[DataDump] Insert graphId:" << graph.graph_id() << " data dumper failed";
  }
}

void AscendKernelRuntime::LaunchDataDump(GraphId graph_id) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    return;
  }
  MS_LOG(INFO) << "Start Launch Dump Data";
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(graph_id);
  if (auto dumper_iter = graph_data_dumper_.find(graph_id); dumper_iter != graph_data_dumper_.end()) {
    auto &data_dumper = dumper_iter->second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->set_runtime_info(runtime_info_map);
    data_dumper->LoadDumpInfo();
  } else {
    MS_LOG(EXCEPTION) << "GraphId:" << graph_id << " not found";
  }
}
#endif

void AscendKernelRuntime::TaskFailCallback(rtExceptionInfo *task_fail_info) {
  if (task_fail_info == nullptr || current_graph_ == nullptr) {
    MS_LOG(ERROR) << "Execute TaskFailCallback failed. task_fail_info or current_graph_ is nullptr";
    return;
  }

  static std::mutex exception_mutex;
  constexpr uint32_t kOverflowThreshold = 5;
  std::lock_guard<std::mutex> lock(exception_mutex);
  if (task_fail_info->retcode == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    auto node = AscendKernelRuntime::GetErrorNodeName(task_fail_info->streamid, task_fail_info->taskid);
    if (!node) {
      MS_LOG(WARNING) << "Node run task overflow, node name is unknown.";
    } else {
      auto key = std::to_string(task_fail_info->streamid) + std::to_string(task_fail_info->taskid) +
                 std::to_string(current_graph_->graph_id());
      if (overflow_tasks_.find(key) == overflow_tasks_.end() || overflow_tasks_[key] == kOverflowThreshold) {
        // print overflow info
        MS_LOG(WARNING) << "Node run task overflow, node name: " << node->fullname_with_scope()
                        << "Task overflow infos task_id: " << task_fail_info->taskid
                        << ", stream_id: " << task_fail_info->streamid << ", tid: " << task_fail_info->tid
                        << ", device_id: " << task_fail_info->deviceid << ", retcode: " << task_fail_info->retcode
                        << " (" << GetErrorMsg(task_fail_info->retcode) << ")" << trace::DumpSourceLines(node);
        overflow_tasks_[key] = 1;
      } else {
        overflow_tasks_[key]++;
      }
    }
  } else {
    task_fail_infoes_.push_back(*task_fail_info);
  }
}

CNodePtr AscendKernelRuntime::GetErrorNodeName(uint32_t streamid, uint32_t taskid) {
  if (current_graph_ == nullptr) {
    return nullptr;
  }
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(current_graph_->graph_id());
  for (const auto &iter : runtime_info_map) {
    MS_EXCEPTION_IF_NULL(iter.second);
    auto task_id = std::get<kTupleTaskId>(*iter.second);
    auto stream_id = std::get<kTupleStreamId>(*iter.second);
    if (task_id == taskid && stream_id == streamid) {
      auto &execute_node = current_graph_->execution_order();
      auto node = std::find_if(execute_node.begin(), execute_node.end(), [&iter](const auto &node) {
        MS_EXCEPTION_IF_NULL(node);
        return node->UniqueName() == iter.first;
      });
      if (node != execute_node.end()) {
        return *node;
      }
    }
  }
  return nullptr;
}

std::string AscendKernelRuntime::GetDumpPath() {
  uint32_t rank_id = 0;
  auto inst = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  if (inst->parallel_mode() != parallel::kStandalone) {
    if (!CommManager::GetInstance().GetRankID(kHcclWorldGroup, &rank_id)) {
      MS_LOG(WARNING) << "Get rank id failed, now using the default value 0.";
    }
  }

  auto ms_om_path = common::GetEnv("MS_OM_PATH");
  std::string path;
  const auto kSuffix = "/node_dump";
  if (ms_om_path.empty()) {
    MS_LOG(WARNING) << "The environment variable 'MS_OM_PATH' is not set, the files of node dump will save to the "
                    << "process local path, as ./rank_id/node_dump/...";
    path = "./rank_" + std::to_string(rank_id) + kSuffix;
  } else {
    path = ms_om_path + "/rank_" + std::to_string(rank_id) + kSuffix;
  }
  return path;
}

#ifndef ENABLE_SECURITY
void AscendKernelRuntime::DumpTaskExceptionInfo(const session::KernelGraph &) {
  const std::string path = GetDumpPath();
  if (access(path.c_str(), F_OK) == 0) {
    if (!DeleteDumpDir(path)) {
      MS_LOG(ERROR) << "Delete dump directory " << path << " failed";
    }
  }
  for (const auto &task_fail_info : task_fail_infoes_) {
    MS_LOG(ERROR) << "Task fail infos task_id: " << task_fail_info.taskid << ", stream_id: " << task_fail_info.streamid
                  << ", tid: " << task_fail_info.tid << ", device_id: " << task_fail_info.deviceid
                  << ", retcode: " << task_fail_info.retcode << " (" << GetErrorMsg(task_fail_info.retcode) << ")";
    auto node = AscendKernelRuntime::GetErrorNodeName(task_fail_info.streamid, task_fail_info.taskid);
    // Dump error data in local path
    if (node == nullptr) {
      continue;
    }
    auto full_scope_name = node->fullname_with_scope();
    MS_LOG(ERROR) << "Dump node (" << full_scope_name << ") task error input/output data to: " << path
                  << trace::DumpSourceLines(node);

    // full_scope_name: Default/GetNext-op1
    std::string lower_full_scope_name(full_scope_name.length(), ' ');
    (void)std::transform(full_scope_name.begin(), full_scope_name.end(), lower_full_scope_name.begin(), ::tolower);
    if (lower_full_scope_name.find("getnext") != std::string::npos) {
      MS_LOG(WARNING) << "GetNext error may be caused by slow data processing (bigger than 20s / batch) or "
                      << "transfer data to device error.";
      MS_LOG(WARNING) << "Suggestion: ";
      MS_LOG(WARNING) << "    1) Set the parameter dataset_sink_mode=False of model.train(...) or "
                      << "model.eval(...) and try again.";
      MS_LOG(WARNING) << "    2) Reduce the batch_size in data processing and try again.";
      MS_LOG(WARNING) << "    3) You can create iterator by interface create_dict_iterator() of dataset class to "
                      << "independently verify the performance of data processing without training. "
                      << "Refer to the link for data processing optimization suggestions: "
                      << "https://www.luojianet_ms.cn/docs/programming_guide/zh-CN/r1.6/optimize_data_processing.html";
    }

    E2eDump::DumpInputData(node, false, path, &full_scope_name);
    E2eDump::DumpOutputData(node, false, path, &full_scope_name);
  }
}
#endif

bool AscendKernelRuntime::Run(const session::KernelGraph &graph, bool is_task_sink) {
  const uint64_t kUSecondInSecond = 1000000;
  SignalGuard sg(IntHandler);
  bool ret = false;

  if (is_task_sink) {
#if defined(_WIN32) || defined(_WIN64)
    auto start_time = std::chrono::steady_clock::now();
#else
    struct timeval start_time {};
    struct timeval end_time {};
    (void)gettimeofday(&start_time, nullptr);
#endif
    ret = RunTask(graph);
#if defined(_WIN32) || defined(_WIN64)
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, kUSecondInSecond>> cost = end_time - start_time;
    MS_LOG(INFO) << "Call MS Run Success in " << cost.count() << " us";
#else
    (void)gettimeofday(&end_time, nullptr);
    uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
    cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
    MS_LOG(INFO) << "Call MS Run Success in " << cost << " us";
#endif
  } else {
    ret = LaunchKernels(graph);
  }

  return ret;
}

void AscendKernelRuntime::SetKernelModStream(const std::vector<CNodePtr> &kernels,
                                             std::vector<size_t> *last_stream_nodes) {
  std::map<void *, size_t> last_kernel;
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &node = kernels[i];
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
    MS_EXCEPTION_IF_NULL(ascend_kernel_mod);
    auto stream_id = AnfAlgo::GetStreamId(kernels[i]);
    auto iter = stream_id_map_.find(stream_id);
    if (iter == stream_id_map_.end()) {
      void *stream = nullptr;
      auto ret = rtStreamCreate(&stream, 0);
      if (ret != RT_ERROR_NONE) {
        MS_LOG(EXCEPTION) << "create communication stream failed, ret:" << ret;
      }
      stream_id_map_[stream_id] = stream;
      ascend_kernel_mod->set_stream(stream);
    } else {
      ascend_kernel_mod->set_stream(iter->second);
    }
    if (stream_id > 0) {
      last_kernel[stream_id_map_[stream_id]] = i;
    }
  }
  (void)std::transform(last_kernel.begin(), last_kernel.end(), std::back_inserter(*last_stream_nodes),
                       [](const std::pair<void *, size_t> &item) { return item.second; });
}

void AscendKernelRuntime::GetShadowBackendNodeMap(const session::KernelGraph &graph,
                                                  std::map<AnfNodePtr, AnfNodePtr> *shadow_backend_node_map) {
  auto input_nodes = graph.input_nodes();
  for (auto &node : input_nodes) {
    auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(node, graph);
    for (auto &knode : input_nodes) {
      if (knode == node) break;
      if (!common::AnfAlgo::IsTupleOutput(front_node) && front_node != nullptr &&
          front_node == AnfAlgo::FetchFrontNodeByBackendNode(knode, graph)) {
        shadow_backend_node_map->emplace(node, knode);
        break;
      }
    }
  }
}

DeviceAddressPtr AscendKernelRuntime::GetInternalDeviceAddress(const session::KernelGraph &graph,
                                                               const AnfNodePtr &node) {
  auto front_node = graph.GetFrontNodeByInternalParameter(node);
  if (front_node.first == nullptr) {
    return nullptr;
  }
  auto pre_graphs = graph.get_pre_graphs();
  for (auto pre_graph_item : pre_graphs) {
    auto pre_graph = pre_graph_item.second.lock();
    MS_EXCEPTION_IF_NULL(pre_graph);
    auto graph_output = pre_graph->GetGraphOutputByFrontNode(front_node);
    if (graph_output.first == nullptr) {
      continue;
    }
    if (!AnfAlgo::OutputAddrExist(graph_output.first, graph_output.second)) {
      return nullptr;
    }
    auto output_device_address = AnfAlgo::GetMutableOutputAddr(graph_output.first, graph_output.second);
    MS_EXCEPTION_IF_NULL(output_device_address);
    if (output_device_address->DeviceType() == DeviceAddressType::kAscend) {
      return output_device_address;
    }
  }
  return nullptr;
}

void AscendKernelRuntime::GenKernelEvents(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  if (kernels.empty() || graph_kernel_events_map_.find(graph.graph_id()) != graph_kernel_events_map_.end()) {
    return;
  }
  std::vector<size_t> last_stream_nodes;
  SetKernelModStream(kernels, &last_stream_nodes);
  auto kernel_events = std::pair<std::map<AnfNodePtr, std::vector<std::function<void()>>>,
                                 std::map<AnfNodePtr, std::vector<std::function<void()>>>>();
  auto &kernel_pre_run_events = kernel_events.first;
  auto &kernel_post_run_events = kernel_events.second;
  auto stream_num = stream_id_map_.size();
  std::vector<std::vector<bool>> kernel_hit(kernels.size(), std::vector<bool>(stream_num, false));
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &kernel = kernels[i];
    auto curr_stream_id = AnfAlgo::GetStreamId(kernel);
    if (stream_id_map_.find(curr_stream_id) == stream_id_map_.end()) {
      MS_LOG(EXCEPTION) << "Stream " << curr_stream_id << "has not been created";
    }
    auto wait_stream = stream_id_map_[curr_stream_id];
    std::vector<bool> stream_hit(stream_num, false);
    std::vector<AnfNodePtr> used_kernels;
    std::set<AnfNodePtr> visited_kernels;
    common::AnfAlgo::GetAllVisitedCNode(kernel, &used_kernels, &visited_kernels);
    bool found_depend = false;
    for (int k = SizeToInt(i) - 1; k >= 0; --k) {
      auto pre_cnode = kernels[IntToSize(k)];
      auto pre_cnode_stream_id = AnfAlgo::GetStreamId(pre_cnode);
      if (pre_cnode_stream_id == curr_stream_id) {
        found_depend = true;
        continue;
      }
      for (auto &visited : used_kernels) {
        if (visited == pre_cnode && !stream_hit[pre_cnode_stream_id] && !kernel_hit[IntToSize(k)][curr_stream_id]) {
          stream_hit[pre_cnode_stream_id] = true;
          kernel_hit[IntToSize(k)][curr_stream_id] = true;
          found_depend = true;
          auto record_stream = stream_id_map_[pre_cnode_stream_id];
          auto event = CreateDeviceEvent();
          event->set_wait_stream(wait_stream);
          event->set_record_stream(record_stream);
          kernel_post_run_events[pre_cnode].emplace_back([event]() { event->RecordEvent(); });
          kernel_pre_run_events[kernel].emplace_back([event]() { event->WaitEvent(); });
        }
      }
    }
    if (!found_depend && wait_stream != stream_) {
      auto pre_event = CreateDeviceEvent();
      pre_event->set_wait_stream(wait_stream);
      pre_event->set_record_stream(stream_);
      kernel_pre_run_events[kernel].emplace_back([pre_event]() { pre_event->RecordEvent(); });
      kernel_pre_run_events[kernel].emplace_back([pre_event]() { pre_event->WaitEvent(); });
    }
  }
  ProcessBoundaryEvent(kernels, &kernel_post_run_events, last_stream_nodes);
  graph_kernel_events_map_[graph.graph_id()] = std::move(kernel_events);
}

void AscendKernelRuntime::GenKernelEventsForMindRT(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  if (kernels.empty() || graph_kernel_events_map_.find(graph.graph_id()) != graph_kernel_events_map_.end()) {
    return;
  }
  std::vector<size_t> last_stream_nodes;
  SetKernelModStream(kernels, &last_stream_nodes);
  auto kernel_events = std::pair<std::map<AnfNodePtr, std::vector<std::function<void()>>>,
                                 std::map<AnfNodePtr, std::vector<std::function<void()>>>>();
  auto &kernel_pre_run_events = kernel_events.first;
  auto &kernel_post_run_events = kernel_events.second;
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &kernel = kernels[i];
    auto curr_stream_id = AnfAlgo::GetStreamId(kernel);
    if (stream_id_map_.find(curr_stream_id) == stream_id_map_.end()) {
      MS_LOG(EXCEPTION) << "Stream " << curr_stream_id << "has not been created.";
    }
    auto wait_stream = stream_id_map_[curr_stream_id];
    std::vector<AnfNodePtr> used_kernels;
    std::set<AnfNodePtr> visited_kernels;
    common::AnfAlgo::GetAllVisitedCNode(kernel, &used_kernels, &visited_kernels);
    bool found_depend = false;
    std::set<AnfNodePtr> record_nodes;
    // set events for nodes and its input: [input_node_stream, node_stream]
    for (auto &visited : used_kernels) {
      auto pre_cnode_stream_id = AnfAlgo::GetStreamId(visited);
      if (stream_id_map_.find(pre_cnode_stream_id) == stream_id_map_.end()) {
        MS_LOG(EXCEPTION) << "Stream " << pre_cnode_stream_id << "has not been created.";
      }
      if (pre_cnode_stream_id == curr_stream_id) {
        found_depend = true;
        continue;
      }
      if (record_nodes.find(visited) == record_nodes.end()) {
        found_depend = true;
        auto record_stream = stream_id_map_[pre_cnode_stream_id];
        auto event = CreateDeviceEvent();
        event->set_wait_stream(wait_stream);
        event->set_record_stream(record_stream);
        kernel_post_run_events[visited].emplace_back([event]() { event->RecordEvent(); });
        kernel_pre_run_events[kernel].emplace_back([event]() { event->WaitEvent(); });
      }
      record_nodes.insert(visited);
    }
    // for start_node(no inputs), set event [stream_, start_node_stream]
    if (!found_depend && wait_stream != stream_) {
      auto pre_event = CreateDeviceEvent();
      pre_event->set_wait_stream(wait_stream);
      pre_event->set_record_stream(stream_);
      kernel_pre_run_events[kernel].emplace_back([pre_event]() { pre_event->RecordEvent(); });
      kernel_pre_run_events[kernel].emplace_back([pre_event]() { pre_event->WaitEvent(); });
    }
  }
  // find end node of graph by last_stream_nodes, and set event [last_node_stream, stream_]
  ProcessBoundaryEvent(kernels, &kernel_post_run_events, last_stream_nodes);
  graph_kernel_events_map_[graph.graph_id()] = std::move(kernel_events);
}

std::pair<vector<std::function<void()>>, vector<std::function<void()>>> AscendKernelRuntime::GetKernelEventFuncs(
  const CNodePtr &kernel) const {
  std::map<AnfNodePtr, std::vector<std::function<void()>>> kernels_pre_event_funcs;
  std::map<AnfNodePtr, std::vector<std::function<void()>>> kernels_post_event_funcs;
  std::vector<std::function<void()>> kernel_pre_event_funcs;
  std::vector<std::function<void()>> kernel_post_event_funcs;

  auto graph_id = AnfAlgo::GetGraphId(kernel.get());
  auto events_iter = graph_kernel_events_map_.find(graph_id);
  if (events_iter != graph_kernel_events_map_.end()) {
    kernels_pre_event_funcs = events_iter->second.first;
    kernels_post_event_funcs = events_iter->second.second;
  }

  auto pre_event_funcs_iter = kernels_pre_event_funcs.find(kernel);
  if (pre_event_funcs_iter != kernels_pre_event_funcs.end()) {
    kernel_pre_event_funcs = pre_event_funcs_iter->second;
  }

  auto post_event_funcs_iter = kernels_post_event_funcs.find(kernel);
  if (post_event_funcs_iter != kernels_post_event_funcs.end()) {
    kernel_post_event_funcs = post_event_funcs_iter->second;
  }

  return std::make_pair(kernel_pre_event_funcs, kernel_post_event_funcs);
}

void AscendKernelRuntime::ProcessBoundaryEvent(
  const std::vector<CNodePtr> &kernels, std::map<AnfNodePtr, std::vector<std::function<void()>>> *kernel_run_events,
  const std::vector<size_t> &last_stream_nodes) {
  for (auto &i : last_stream_nodes) {
    if (i >= kernels.size()) {
      MS_LOG(ERROR) << "Node index exceed kernel size.";
      continue;
    }
    auto &kernel = kernels[i];
    MS_EXCEPTION_IF_NULL(kernel);
    bool found_nearest_child = false;
    for (size_t j = i + 1; j < kernels.size(); ++j) {
      auto &child = kernels[j];
      MS_EXCEPTION_IF_NULL(child);
      auto input_size = child->inputs().size() - 1;
      for (size_t k = 0; k < input_size; ++k) {
        auto kernel_index =
          common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(child, k), 0, true);
        if (kernel_index.first == kernel) {
          found_nearest_child = true;
          break;
        }
      }
      if (found_nearest_child) {
        break;
      }
    }
    if (!found_nearest_child) {
      auto post_event = CreateDeviceEvent();
      MS_EXCEPTION_IF_NULL(post_event);
      auto id = AnfAlgo::GetStreamId(kernel);
      auto record_stream = stream_id_map_[id];
      post_event->set_wait_stream(stream_);
      post_event->set_record_stream(record_stream);
      (*kernel_run_events)[kernel].emplace_back([post_event]() { post_event->RecordEvent(); });
      (*kernel_run_events)[kernel].emplace_back([post_event]() { post_event->WaitEvent(); });
    }
  }
}

bool AscendKernelRuntime::RunDynamicKernelAsync(const session::KernelGraph &graph) {
  MS_LOG(INFO) << "RunExecutorAsync start. GraphId:" << graph.graph_id();
  auto iter = graph_dynamic_kernel_map_.find(graph.graph_id());
  if (iter == graph_dynamic_kernel_map_.end()) {
    MS_LOG(ERROR) << "GraphId:" << graph.graph_id() << " Not Found! Please generator executor first";
    return false;
  }
  AscendEnableDynamicRuntimeCache(&graph);

  auto dynamic_kernels = iter->second;
  for (const auto &dynamic_kernel : dynamic_kernels) {
    MS_EXCEPTION_IF_NULL(dynamic_kernel);
    if (dynamic_kernel->have_depends() || dynamic_kernel->GetKernelType() == KernelType::HCCL_KERNEL) {
      MS_LOG(INFO) << "Match Dynamic Kernel, Start SyncStream";
      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return false;
      }
    }

    if (dynamic_kernel->is_dynamic_shape()) {
      dynamic_kernel->InferShape();
      dynamic_kernel->UpdateArgs();
    }

    dynamic_kernel->Execute();
    dynamic_kernel->PostExecute();
  }

  if (!SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed";
    return false;
  }

  return true;
}

bool AscendKernelRuntime::RunTask(const session::KernelGraph &graph) {
  current_graph_ = &graph;
  SetCurrentContext();
  if (graph.is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic Shape Graph Run Task Async";
    return RunDynamicKernelAsync(graph);
  }

  MS_LOG(INFO) << "RunTask start. GraphId:" << graph.graph_id();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (GraphWithEmptyTaskList(graph)) {
    MS_LOG(INFO) << "RunTask end, no task info found";
    return true;
  }

  if (!CheckGraphIdValid(graph.graph_id())) {
    MS_LOG(ERROR) << "GraphId:" << graph.graph_id() << " Invalid! Graph RunTask without GenTask.";
    return false;
  }

  try {
    ModelRunner::Instance().RunModel(graph.graph_id());
  } catch (const std::exception &) {
#ifndef ENABLE_SECURITY
    DumpTaskExceptionInfo(graph);
#endif
#ifdef ENABLE_TDTQUE
    // Run task error, we should call TdtHostDestroy to release tdt to avoid DeviceQueueOp hostPush hung
    // case1: cpu usage 100% cause thread/process exit, but some tdt thread remain in backend
    if (!TdtHandle::DestroyHandle()) {
      MS_LOG(WARNING) << "Destroy tdt channel failed.";
    } else {
      MS_LOG(INFO) << "Destroy tdt channel success.";
    }
#endif
    return false;
  }
  task_fail_infoes_.clear();
  return true;
}

bool AscendKernelRuntime::SyncStream() {
  SetCurrentContext();
  session::PynativeTaskManager::GetInstance().ExecuteRemainingTasks();
  for (auto &iter : stream_id_map_) {
    if (rtStreamSynchronize(iter.second) != RT_ERROR_NONE) {  // o for switch stream
      MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
      return false;
    }
  }
  return true;
}

bool AscendKernelRuntime::MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) {
  SetCurrentContext();
  if (stream_ == nullptr) {
    MS_LOG(ERROR) << "MemcpyAsync failed. stream_ is nullptr";
    return false;
  }

  auto copy_kind = static_cast<rtMemcpyKind_t>(kind);
  if (copy_kind != RT_MEMCPY_HOST_TO_DEVICE_EX && copy_kind != RT_MEMCPY_DEVICE_TO_DEVICE) {
    MS_LOG(EXCEPTION) << "Memory copy async not support cache host buffer in kind: " << kind;
  }
  if (dst == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync dst ptr is null, copy kind:" << kind;
    return false;
  }
  if (src == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync src ptr is null, copy kind:" << kind;
    return false;
  }
  if (size == 0) {
    MS_LOG(ERROR) << "rtMemcpyAsync size is 0, copy kind:" << kind;
    return false;
  }
  if (RT_ERROR_NONE != rtMemcpyAsync(dst, size, src, size, static_cast<rtMemcpyKind_t>(kind), stream_)) {
    MS_LOG(ERROR) << "Call runtime rtMemcpyAsync error.";
    return false;
  }
  return true;
}

void AscendKernelRuntime::CreateContext() {
  if (rt_context_ == nullptr) {
    auto ret = rtCtxCreate(&rt_context_, 0, UintToInt(device_id_));
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call rtCtxCreate, ret[" << static_cast<int>(ret) << "]";
    }
  }
  SetCurrentContext();
}

bool AscendKernelRuntime::InitDevice() {
  int device_count = 0;
  auto ret = rtGetDeviceCount(&device_count);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }

  ret = rtSetDevice(UintToInt(device_id_));
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr == nullptr) {
    MS_LOG(ERROR) << "Get MsContext instance failed";
    return false;
  }
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    if (!HcclInit()) {
      MS_LOG(ERROR) << "HcclInit init failed";
      return false;
    }
  }

  // Context will be created by rtSetDevice
  ret = rtCtxGetCurrent(&rt_context_);
  if (ret != RT_ERROR_NONE || rt_context_ == nullptr) {
    MS_LOG(ERROR) << "Call rtCtxGetCurrent failed, ret[" << ret << "]";
    return false;
  }

  ret = rtStreamCreateWithFlags(&stream_, 0, RT_STREAM_HUGE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtStreamCreate, ret[" << ret << "]";
  }
  ret = rtStreamCreateWithFlags(&independent_stream_, 0, RT_STREAM_HUGE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtStreamCreate, ret[" << ret << "]";
  }
  ret = rtStreamCreate(&communication_stream_, 0);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "create communication stream failed, ret:" << ret;
  }

  stream_id_map_[kDefaultStreamIndex] = stream_;
  stream_id_map_[kIndependentStreamIndex] = independent_stream_;
  stream_id_map_[kWorldGroupStreamIndex] = communication_stream_;
  return true;
}

bool AscendKernelRuntime::ResetDevice(uint32_t device_id) {
  SetCurrentContext();
  int32_t ret;
  for (auto &iter : stream_id_map_) {
    ret = rtStreamDestroy(iter.second);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rtStreamDestroy, ret[" << ret << "]";
    }
    iter.second = nullptr;
  }
  ret = rtDeviceReset(UintToInt(device_id));
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtDeviceReset, ret[" << ret << "]";
  }
  // set to nullptr as its not created, only bounded to existing context
  rt_context_ = nullptr;
  return true;
}

bool AscendKernelRuntime::HcclInit() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context::IsTsdOpened(context_ptr)) {
    MS_LOG(EXCEPTION) << "Hccl dependent tsd is not open";
  }
  MS_LOG(INFO) << "Do hcom init";
  bool is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (!is_task_sink && mode == kGraphMode) {
    (void)hccl::HcclAdapter::GetInstance().InitHccl();
    auto rank_size = HcclCollectiveGroup::instance().GetRankSize();
    std::vector<unsigned int> ranks(rank_size);
    std::iota(std::begin(ranks), std::end(ranks), 0);
    HcclCollectiveGroup::instance().CreateCommGroup(kHcclWorldGroup, ranks);
    return true;
  }

  auto config_path_str = std::getenv("LUOJIANET_MS_HCCL_CONFIG_PATH");
  if (config_path_str == nullptr) {
    config_path_str = std::getenv("RANK_TABLE_FILE");
    if (config_path_str == nullptr) {
      MS_LOG(ERROR) << "The environment variable 'LUOJIANET_MS_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE' is not set, so get"
                    << " hccl json config failed, please set env 'LUOJIANET_MS_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE'";
      return false;
    }
  }
  if (strlen(config_path_str) >= kPathMax) {
    MS_LOG(ERROR) << "Invalid environment variable 'LUOJIANET_MS_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE', the path length"
                  << " should be smaller than " << kPathMax << ", but got " << config_path_str;
    return false;
  }
  std::string rank_id_str = GetRankIdStr();
  auto full_path = realpath(config_path_str, nullptr);
  if (full_path == nullptr) {
    MS_LOG(ERROR) << "Invalid environment variable 'LUOJIANET_MS_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE', the path is: "
                  << config_path_str << ". Please check (1) whether the path exists, "
                  << "(2) whether the path has the access permission, (3) whether the path is too long. ";
    return false;
  }
  MS_LOG(INFO) << "LUOJIANET_MS_HCCL_CONFIG_PATH : " << full_path << ", RANK_ID: " << rank_id_str;
  bool ret = hccl::HcclAdapter::GetInstance().InitHccl(
    context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID), rank_id_str, full_path,
    mode == kGraphMode ? hccl::HcclMode::kGraph : hccl::HcclMode::kPynative);
  free(full_path);
  if (!ret) {
    MS_LOG(ERROR) << "Hcom init failed.";
    return false;
  }
  return true;
}

bool AscendKernelRuntime::DestroyHccl() {
  if (!NeedDestroyHccl()) {
    MS_LOG(INFO) << "Hccl is not enable, no need to close.";
    return true;
  }
  bool res = hccl::HcclAdapter::GetInstance().FinalizeHccl();
  if (!res) {
    MS_LOG(ERROR) << "Hccl destroy failed";
    return false;
  }
  MS_LOG(INFO) << "Hccl destroy successful.";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  context_ptr->set_param<bool>(MS_CTX_ENABLE_HCCL, false);
  return true;
}

bool AscendKernelRuntime::GraphWithEmptyTaskList(const session::KernelGraph &graph) const {
  auto iter = task_map_.find(graph.graph_id());
  if (iter == task_map_.end()) {
    MS_LOG(EXCEPTION) << "Unknown graph ptr";
  }
  return iter->second.empty();
}

bool AscendKernelRuntime::CheckGraphIdValid(GraphId graph_id) const {
  return task_map_.find(graph_id) != task_map_.end() && graph_model_map_.find(graph_id) != graph_model_map_.end();
}

void AscendKernelRuntime::KernelLaunchProfiling(const std::string &kernel_name) {
#ifndef ENABLE_SECURITY
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  if (!profiler_manager->GetProfilingEnableFlag()) {
    return;
  }

  // save task info
  uint32_t stream_id;
  uint32_t task_id;
  auto rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Profiling get task_id stream_id failed";
  }
  std::pair<uint32_t, uint32_t> stream_task_pair = {stream_id, task_id};
  auto try_emplace_ret = stream_id_task_id_op_name_map_.try_emplace(stream_task_pair, kernel_name);
  if (!try_emplace_ret.second) {
    MS_LOG(WARNING) << "Profiling duplicate key, task_id:" << stream_task_pair.second
                    << " stream_id:" << stream_task_pair.first << " name:" << kernel_name;
  }
  if (stream_id_task_id_op_name_map_.size() > kProfilingMaxTaskIdInStream) {
    MS_LOG(EXCEPTION) << "Too many profiling data";
  }
#endif
}

std::shared_ptr<DeviceEvent> AscendKernelRuntime::CreateDeviceEvent() {
  auto ascend_event = std::make_shared<AscendEvent>();
  MS_EXCEPTION_IF_NULL(ascend_event);
  return ascend_event;
}

std::shared_ptr<DeviceEvent> AscendKernelRuntime::CreateDeviceTimeEvent() {
  auto ascend_time_event = std::make_shared<AscendTimeEvent>();
  MS_EXCEPTION_IF_NULL(ascend_time_event);
  return ascend_time_event;
}

uint64_t AscendKernelRuntime::GetAvailableMemMaxSize() const {
  auto ascend_mem_manager = std::dynamic_pointer_cast<AscendMemoryManager>(mem_manager_);
  MS_EXCEPTION_IF_NULL(ascend_mem_manager);
  return ascend_mem_manager->GetMsMaxMemSize();
}

uint64_t AscendKernelRuntime::GetMsUsedHbmSize() const {
  auto ascend_mem_manager = std::dynamic_pointer_cast<AscendMemoryManager>(mem_manager_);
  MS_EXCEPTION_IF_NULL(ascend_mem_manager);
  return ascend_mem_manager->GetMsUsedHbmSize();
}

bool AscendKernelRuntime::DeleteDumpDir(const std::string &path) {
  string real_path = GetRealPath(path);
  if (DeleteDumpFile(real_path) == -1) {
    return false;
  }
  if (rmdir(real_path.c_str()) == -1) {
    MS_LOG(WARNING) << "Delete dir " << real_path << " failed!";
  }
  return true;
}

int AscendKernelRuntime::DeleteDumpFile(std::string path) {
  DIR *dir;
  struct dirent *dirinfo;
  struct stat statbuf;
  string filepath;
  int result = 0;
  lstat(path.c_str(), &statbuf);

  if (S_ISREG(statbuf.st_mode)) {
    result = remove(path.c_str());
  } else if (S_ISDIR(statbuf.st_mode)) {
    if ((dir = opendir(path.c_str())) == nullptr) {
      return -1;
    }

    while (!result && (dirinfo = readdir(dir))) {
      if (path[path.size() - 1] != '/') {
        path = path + "/";
      }
      MS_EXCEPTION_IF_NULL(dirinfo);
      filepath = path + dirinfo->d_name;
      if (strcmp(dirinfo->d_name, ".") == 0 || strcmp(dirinfo->d_name, "..") == 0) continue;
      result = DeleteDumpFile(filepath);
      if (!result) {
        if (rmdir(filepath.c_str()) == -1) {
          MS_LOG(WARNING) << "Delete dir " << filepath << " failed!";
        }
      }
    }
    if (closedir(dir) == -1) {
      MS_LOG(WARNING) << "Dump dir " << path << " close failed!";
    }
  }
  return result;
}

std::string AscendKernelRuntime::GetRealPath(const std::string &path) {
  char real_path_mem[kPathMax] = {0};
  char *real_path_ret = realpath(path.c_str(), real_path_mem);
  if (real_path_ret == nullptr) {
    return "";
  }
  return std::string(real_path_mem);
}

void AscendKernelRuntime::SetReuseCommunicationAddress(const session::KernelGraph &graph) {
  auto cnode_list = graph.execution_order();
  for (const auto &cnode : cnode_list) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (common::AnfAlgo::HasNodeAttr(kAttrReuseCommunication, cnode)) {
      auto reuse_index = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrReuseCommunication);
      if (reuse_communication_address_.find(reuse_index) == reuse_communication_address_.end()) {
        (void)reuse_communication_address_.emplace(reuse_index, std::make_pair(nullptr, nullptr));
      }
    }
  }
}
}  // namespace luojianet_ms::device::ascend
