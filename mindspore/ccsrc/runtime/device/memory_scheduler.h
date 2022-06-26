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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <utility>
#include "runtime/device/memory_offload_strategy.h"

namespace mindspore {
namespace device {
class MemHandler {
 public:
  MemHandler() = default;
  virtual ~MemHandler() = default;
  virtual size_t GetAvailableMemSize() = 0;
  virtual void *MallocDevice(size_t mem_size) = 0;
  virtual void FreeDevice(void *ptr) = 0;
  virtual void *MallocHost(size_t mem_size) = 0;
  virtual void FreeHost(void *ptr) = 0;
  virtual void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) = 0;
  virtual void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) = 0;
};

class MemScheduler {
 public:
  MemScheduler() = default;
  ~MemScheduler() = default;

  bool need_record_event() const { return need_record_event_; }

  void set_need_record_event(bool flag) { need_record_event_ = flag; }

  bool optimized() const { return optimized_; }

  void Update();

  void SetMemHandler(const std::shared_ptr<MemHandler> &handler) { mem_handler_ = handler; }

  void Init(const void *key, void *host_ptr, size_t mem_size, MemPriority priority = kMemPriorityLow);

  void *GetOrMalloc(const void *key, size_t mem_size, MemPriority priority = kMemPriorityLow);

  bool HasDeviceMem(const void *key) const { return mem_result_.find(key) != mem_result_.end(); }

  void UpdateHighPriorityMem(const void *key) {
    if (need_record_event_) {
      (void)high_priority_updated_step_[key].emplace_back(current_step_);
    }
  }

  void SetTotalStep(size_t step) {
    total_step_ = step;
    step_keys_.resize(total_step_);
  }

  void Reset() { current_step_ = 0; }

  bool PreCompute(void *stream);

  bool PostCompute(void *stream);

  bool Optimize();

  void Clear();

  void ClearAllocatedMem();

  void SetOffload(const void *key) { (void)manual_offload_keys_.insert(key); }

  void AddMemNeedInit(const void *key) { (void)high_priority_mem_need_init_.insert(key); }

  void ClearMemNeedInit() { high_priority_mem_need_init_.clear(); }

 private:
  void Record(const void *key, const MemEventType &event_type, size_t mem_size = 0);

  void OptMemUsage(float mem_used_factor = 1.0f);

  bool Mock();

  void AdjustFirstEventIndex();

  void *MallocDevice(size_t mem_size, void *stream);

  void SwapOutAndFreeDevice(const void *key, void *device_ptr, size_t mem_size, void *stream);

  size_t GetMemSize(const void *key);

  void *GetOrMallocHostPtr(const void *key, size_t mem_size);

  void GetHostPtr(const void *key, void **host_ptr, bool *from_init);

  bool PreComputeInit(const std::shared_ptr<MemEvent> &event, void *stream);

  bool PreComputeMalloc(const std::shared_ptr<MemEvent> &event, void *stream);

  bool PreComputeSwapIn(const std::shared_ptr<MemEvent> &event, void *stream);

  bool PreComputeGet(const std::shared_ptr<MemEvent> &event, void *stream);

  std::map<const void *, MemPriority> mem_priority_;
  std::map<const void *, std::vector<std::shared_ptr<MemEvent>>> mem_events_;
  std::set<const void *> manual_offload_keys_;
  std::vector<std::set<const void *>> step_keys_;
  std::map<const void *, void *> mem_result_;
  std::map<const void *, void *> init_host_ptr_;
  std::map<const void *, void *> swap_host_ptr_;
  std::map<const void *, std::vector<size_t>> high_priority_updated_step_;
  std::set<const void *> high_priority_mem_need_init_;
  size_t total_step_{0};
  size_t current_step_{0};
  bool need_record_event_{true};
  bool optimized_{false};
  double compute_start_time_{0};
  std::vector<double> compute_time_;
  bool record_compute_time_{false};
  bool updated_{false};
  std::shared_ptr<MemHandler> mem_handler_{nullptr};
  std::shared_ptr<MemOffloadStrategy> strategy_{nullptr};
};

class MemSchedulerManager {
 public:
  MemSchedulerManager() = default;
  ~MemSchedulerManager() = default;
  std::shared_ptr<MemScheduler> GetOrCreateMemScheduler(uint64_t uid) {
    auto scheduler = GetMemScheduler(uid);
    if (scheduler == nullptr) {
      scheduler = std::make_shared<MemScheduler>();
      graph_mem_scheduler_map_[uid] = scheduler;
    }
    return scheduler;
  }

  std::shared_ptr<MemScheduler> GetMemScheduler(uint64_t uid) {
    auto iter = graph_mem_scheduler_map_.find(uid);
    if (iter != graph_mem_scheduler_map_.end()) {
      return iter->second;
    }
    return nullptr;
  }

 private:
  std::map<uint64_t, std::shared_ptr<MemScheduler>> graph_mem_scheduler_map_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
