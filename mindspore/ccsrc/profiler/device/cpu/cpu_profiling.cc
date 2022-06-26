/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "profiler/device/cpu/cpu_profiling.h"

#include <cxxabi.h>
#include <cmath>
#include <ctime>
#include "profiler/device/cpu/cpu_data_saver.h"
#include "include/common/pybind_api/api_register.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace profiler {
namespace cpu {
std::shared_ptr<CPUProfiler> CPUProfiler::profiler_inst_ = std::make_shared<CPUProfiler>();

std::shared_ptr<CPUProfiler> &CPUProfiler::GetInstance() { return profiler_inst_; }

void CPUProfiler::Init(const std::string &profileDataPath = "") {
  MS_LOG(INFO) << "Initialize CPU Profiling";
  base_time_ = GetHostMonoTimeStamp();
  profile_data_path_ = profileDataPath;
  MS_LOG(INFO) << " Host start time(ns): " << base_time_ << " profile data path: " << profile_data_path_;
}

void CPUProfiler::StepProfilingEnable(const bool enable_flag) {
  MS_LOG(INFO) << "CPU Profiler enable flag: " << enable_flag;
  enable_flag_ = enable_flag;
}

void CPUProfiler::SetRunTimeData(const std::string &op_name, const uint32_t pid, bool is_parallel) {
  if (!is_parallel) {
    op_name_ = op_name;
    pid_ = pid;
  }
  {
    std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
    auto iter = op_info_map_.find(op_name);
    if (iter != op_info_map_.end()) {
      iter->second.op_count += 1;
      return;
    }
  }
  std::unique_lock<std::shared_mutex> lock(op_map_mutex_);
  OpInfo op_info;
  op_info.op_name = op_name;
  op_info.pid = pid;
  op_info.op_count = 1;
  op_info_map_[op_name] = op_info;
}

void CPUProfiler::SetRuntimeStart(const std::string op_name, const uint64_t start_timestamp) {
  std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.tmp_start_duration.start_timestamp = start_timestamp;
    auto actor_manager = ActorMgr::GetActorMgrRef();
    MS_EXCEPTION_IF_NULL(actor_manager);
    auto thread_pool = actor_manager->GetActorThreadPool();
    auto worker_ids_map = thread_pool->GetWorkerIdMap();
    auto id_iter = worker_ids_map.find(std::this_thread::get_id());
    if (id_iter != worker_ids_map.end()) {
      iter->second.tmp_start_duration.tid = id_iter->second;
    }
  }
}

float CPUProfiler::SetRuntimeEnd(const std::string op_name, const uint64_t stop_timestamp) {
  float op_time_elapsed = 0;
  std::shared_lock<std::shared_mutex> lock(op_map_mutex_);
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.tmp_start_duration.duration =
      (stop_timestamp - iter->second.tmp_start_duration.start_timestamp) / kNanosecondToMillisecond;
    auto actor_manager = ActorMgr::GetActorMgrRef();
    MS_EXCEPTION_IF_NULL(actor_manager);
    auto thread_pool = actor_manager->GetActorThreadPool();
    auto worker_ids_map = thread_pool->GetWorkerIdMap();
    auto id_iter = worker_ids_map.find(std::this_thread::get_id());
    if (id_iter != worker_ids_map.end()) {
      if (iter->second.tmp_start_duration.tid != id_iter->second) {
        MS_LOG(EXCEPTION) << "Op " << op_name << " start time thread id must be equal to end thread id.";
      }
    }
    (void)iter->second.start_duration.emplace_back(iter->second.tmp_start_duration);
    op_time_elapsed = iter->second.tmp_start_duration.duration;
  }
  return op_time_elapsed;
}

void CPUProfiler::OpDataProducerBeginParallel(const std::string op_name, const uint32_t pid) {
  auto start_timestamp = GetHostMonoTimeStamp();
  SetRunTimeData(op_name, pid, true);
  SetRuntimeStart(op_name, start_timestamp);

#if ENABLE_GPU
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    // For heterogeneous scene, record op name to gpu_profiler_inst.
    auto gpu_profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
    // For cpu network, no gpu profiler, do not to raise exception.
    if (gpu_profiler_inst && gpu_profiler_inst->GetEnableFlag()) {
      gpu_profiler_inst->RecordOneStepStartEndInfo(op_name);
    }
  }
#endif
}

void CPUProfiler::OpDataProducerEndParallel(const std::string op_name) {
  auto stop_timestamp = GetHostMonoTimeStamp();
  float op_time_elapsed = SetRuntimeEnd(op_name, stop_timestamp);
  MS_LOG(DEBUG) << "Host Time Elapsed(ms)," << op_name << "," << op_time_elapsed;
  Profiler::SetRunTimeData(op_name, op_time_elapsed);
}

void CPUProfiler::OpDataProducerBegin(const std::string op_name, const uint32_t pid) {
  op_time_start_ = GetHostMonoTimeStamp();
  op_time_mono_start_ = GetHostMonoTimeStamp();
  SetRunTimeData(op_name, pid);

#if ENABLE_GPU
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    // For heterogeneous scene, record op name to gpu_profiler_inst.
    auto gpu_profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
    // For cpu network, no gpu profiler, do not to raise exception.
    if (gpu_profiler_inst && gpu_profiler_inst->GetEnableFlag()) {
      gpu_profiler_inst->RecordOneStepStartEndInfo(op_name);
    }
  }
#endif
}

void CPUProfiler::OpDataProducerEnd() {
  float op_time_elapsed = 0;
  op_time_stop_ = GetHostMonoTimeStamp();
  op_time_elapsed = (op_time_stop_ - op_time_start_) / kNanosecondToMillisecond;
  MS_LOG(DEBUG) << "Host Time Elapsed(ms)," << op_name_ << "," << op_time_elapsed;
  Profiler::SetRunTimeData(op_name_, op_time_elapsed);
  Profiler::SetRunTimeData(op_name_, op_time_mono_start_, op_time_elapsed);
}

void CPUProfiler::Stop() {
  MS_LOG(INFO) << "Stop CPU Profiling";
  SaveProfileData();
  ClearInst();
}

void CPUProfiler::SaveProfileData() {
  if (profile_data_path_.empty()) {
    MS_LOG(WARNING) << "Profile data path is empty, skip save profile data.";
  } else {
    auto cpu_data_saver_inst = profiler::cpu::CpuDataSaver::GetInstance();
    MS_EXCEPTION_IF_NULL(cpu_data_saver_inst);
    cpu_data_saver_inst->ParseOpInfo(op_info_map_);
    cpu_data_saver_inst->WriteFile(profile_data_path_);
  }
}

void CPUProfiler::ClearInst() { op_info_map_.clear(); }

REGISTER_PYBIND_DEFINE(CPUProfiler_, ([](const py::module *m) {
                         (void)py::class_<CPUProfiler, std::shared_ptr<CPUProfiler>>(*m, "CPUProfiler")
                           .def_static("get_instance", &CPUProfiler::GetInstance, "CPUProfiler get_instance.")
                           .def("init", &CPUProfiler::Init, py::arg("profile_data_path"), "init")
                           .def("stop", &CPUProfiler::Stop, "stop")
                           .def("step_profiling_enable", &CPUProfiler::StepProfilingEnable, py::arg("enable_flag"),
                                "enable or disable step profiling");
                       }));
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore
