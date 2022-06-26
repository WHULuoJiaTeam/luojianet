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

#ifndef MINDSPORE_CCSRC_PROFILER_DEVICE_CPU_PROFILING_H
#define MINDSPORE_CCSRC_PROFILER_DEVICE_CPU_PROFILING_H
#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "profiler/device/profiling.h"
#if ENABLE_GPU
#include "profiler/device/gpu/gpu_profiling.h"
#endif
#include "actor/actormgr.h"

namespace mindspore {
namespace profiler {
namespace cpu {
const float kNanosecondToMillisecond = 1000000;

class CPUProfiler : public Profiler {
 public:
  static std::shared_ptr<CPUProfiler> &GetInstance();
  CPUProfiler() = default;
  ~CPUProfiler() = default;
  CPUProfiler(const CPUProfiler &) = delete;
  CPUProfiler &operator=(const CPUProfiler &) = delete;

  void Init(const std::string &profileDataPath) override;
  void Stop() override;
  void StepProfilingEnable(const bool enable_flag) override;
  void OpDataProducerBegin(const std::string op_name, const uint32_t pid);
  void OpDataProducerEnd() override;
  void OpDataProducerEndParallel(const std::string op_name);
  void OpDataProducerBeginParallel(const std::string op_name, const uint32_t pid);
  float SetRuntimeEnd(const std::string op_name, const uint64_t stop_timestamp);
  void SetRuntimeStart(const std::string op_name, const uint64_t start_timestamp);

 private:
  void SetRunTimeData(const std::string &op_name, const uint32_t pid, bool is_parallel = false);
  void SaveProfileData() override;
  void ClearInst() override;

  static std::shared_ptr<CPUProfiler> profiler_inst_;
  uint64_t base_time_;
  std::string op_name_;
  uint32_t pid_;

  uint64_t op_time_start_;
  uint64_t op_time_mono_start_;
  uint64_t op_time_stop_;
};
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PROFILER_DEVICE_CPU_PROFILING_H
