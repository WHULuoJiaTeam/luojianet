/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef _MSC_VER
#include <sched.h>
#include <unistd.h>
#endif
#include "thread/threadpool.h"
#include "thread/core_affinity.h"

namespace mindspore {
Worker::~Worker() {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    alive_ = false;
  }
  cond_var_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
  pool_ = nullptr;
#ifdef OPERATOR_PARALLELISM
  if (task_messages_ != nullptr) {
    free(task_messages_);
    task_messages_ = nullptr;
  }
#endif
}

void Worker::CreateThread() { thread_ = std::thread(&Worker::Run, this); }

void Worker::SetAffinity() {
#ifdef _WIN32
  SetWindowsSelfAffinity(core_id_);
#elif defined(BIND_CORE)
#ifdef __ANDROID__
  int ret = sched_setaffinity(gettid(), sizeof(cpu_set_t), &mask_);
  if (ret != THREAD_OK) {
    THREAD_ERROR("bind thread %d to cpu failed. ERROR %d", gettid(), errno);
  }
  return;
#else
#if !defined(__APPLE__) && !defined(SUPPORT_MSVC)
  int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_);
  if (ret != THREAD_OK) {
    THREAD_ERROR("bind thread %lu to cpu failed. ERROR %d", pthread_self(), errno);
  }
  return;
#endif
#endif
#endif
}

void Worker::InitWorkerMask(const std::vector<int> &core_list, const size_t workers_size) {
#ifdef _WIN32
  static uint32_t windows_core_index = 0;
  core_id_ = windows_core_index++;
#elif defined(BIND_CORE)
#ifdef SERVER_INFERENCE
  if (core_list.empty()) {
    return;
  }
#endif
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (core_list.size() > 0) {
    CPU_SET(core_list[workers_size % core_list.size()], &mask);
  }
  this->set_mask(mask);
#endif
  return;
}

void Worker::Run() {
  SetAffinity();
#if !defined(__APPLE__) && !defined(SUPPORT_MSVC)
  static std::atomic_int index = {0};
  (void)pthread_setname_np(pthread_self(), ("KernelThread_" + std::to_string(index++)).c_str());
#endif
#ifdef PLATFORM_86
  // Some CPU kernels need set the flush zero mode to improve performance.
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  while (alive_) {
    if (RunLocalKernelTask()) {
      spin_count_ = 0;
    } else {
      YieldAndDeactive();
    }
#ifdef OPERATOR_PARALLELISM
    if (RunQueueWorkTask()) {
      if (spin_count_ > 0) {
        spin_count_ = 1;
      }
    }
#endif
    if (spin_count_ > max_spin_count_) {
      WaitUntilActive();
      spin_count_ = 1;
    }
  }
}

bool Worker::RunLocalKernelTask() {
  Task *task = task_.load(std::memory_order_consume);
  if (task == nullptr) {
    return false;
  }
  int task_id = task_id_.load(std::memory_order_consume);
  task->status |= task->func(task->content, task_id, lhs_scale_, rhs_scale_);
  task_.store(nullptr, std::memory_order_relaxed);
  (void)++task->finished;
  return true;
}

void Worker::YieldAndDeactive() {
  // deactivate this worker only on the first entry
  if (spin_count_ == 0) {
    THREAD_TEST_TRUE(task_ == nullptr);
    status_.store(kThreadIdle);
  }
  spin_count_++;
  std::this_thread::yield();
}

void Worker::WaitUntilActive() {
  std::unique_lock<std::mutex> _l(mutex_);
  cond_var_.wait(_l, [&] { return status_ == kThreadBusy || active_num_ > 0 || !alive_; });
  active_num_--;
}

void Worker::set_scale(float lhs_scale, float rhs_scale) {
  lhs_scale_ = lhs_scale;
  rhs_scale_ = rhs_scale;
}

void Worker::Active(Task *task, int task_id) {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    THREAD_TEST_TRUE(task_ == nullptr);
    task_id_.store(task_id, std::memory_order_relaxed);
    task_.store(task, std::memory_order_release);
    status_ = kThreadBusy;
  }
  cond_var_.notify_one();
}

void Worker::Active() {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    active_num_++;
    status_ = kThreadBusy;
  }
  cond_var_.notify_one();
}

bool Worker::available() {
  int expected = kThreadIdle;
  return status_.compare_exchange_strong(expected, kThreadHeld);
}

bool Worker::check_task_nullptr() {
  std::lock_guard<std::mutex> _l(mutex_);
  if (status_ == kThreadBusy && task_ == nullptr) {
    return true;
  }
  return false;
}

ThreadPool::~ThreadPool() {
  for (auto &worker : workers_) {
    delete worker;
    worker = nullptr;
  }
  workers_.clear();

  if (affinity_ != nullptr) {
    delete affinity_;
    affinity_ = nullptr;
  }
#ifdef OPERATOR_PARALLELISM
#ifdef USE_HQUEUE
  task_queue_.Clean();
#endif
#endif
  THREAD_INFO("destruct success");
}

int ThreadPool::CreateThreads(size_t thread_num, const std::vector<int> &core_list) {
#ifdef OPERATOR_PARALLELISM
#ifdef USE_HQUEUE
  if ((!task_queue_.IsInit()) && task_queue_.Init(MAX_READY_TASK_NR) != true) {
    THREAD_ERROR("Init task queue failed");
    return THREAD_ERROR;
  }
#endif
#endif
  size_t core_num = std::thread::hardware_concurrency();
  thread_num = thread_num < core_num ? thread_num : core_num;
  THREAD_INFO("ThreadInfo, Num: [%zu], CoreNum: [%zu]", thread_num, core_num);
  if (thread_num == 0) {
    THREAD_INFO("Current thread as working thread.");
    return THREAD_OK;
  }
  std::lock_guard<std::mutex> _l(pool_mutex_);
  for (size_t i = 0; i < thread_num; ++i) {
    auto worker = new (std::nothrow) Worker(this);
    THREAD_ERROR_IF_NULL(worker);
    worker->InitWorkerMask(core_list, workers_.size());
    worker->CreateThread();
    workers_.push_back(worker);
    THREAD_INFO("create kernel thread[%zu]", i);
  }
  return THREAD_OK;
}

int ThreadPool::ParallelLaunch(const Func &func, Content content, int task_num) const {
  // if single thread, run master thread
  if (task_num <= 1) {
    for (int i = 0; i < task_num; ++i) {
      int ret = func(content, i, 0, 1);
      if (ret != 0) {
        return ret;
      }
    }
    return THREAD_OK;
  }

  // distribute task to the KernelThread and the idle ActorThread,
  // if the task num is greater than the KernelThread num
  THREAD_DEBUG("launch: %d", task_num);
  Task task = {func, content};
  Worker *curr = CurrentWorker();
  DistributeTask(&task, task_num, curr);
  // synchronization
  // wait until the finished is equal to task_num
  if (curr != nullptr) {
    if (curr->RunLocalKernelTask()) {
      curr->set_task_free(true);
    }
  }
  while (task.finished != task_num) {
#ifdef OPERATOR_PARALLELISM
    if (RunQueueWorkTask() == false && (curr && curr->RunLocalKernelTask() == false)) {
      std::this_thread::yield();
    }
#else
    std::this_thread::yield();
#endif
  }
  // check the return value of task
  if (task.status != THREAD_OK) {
    return THREAD_ERROR;
  }
  return THREAD_OK;
}

void ThreadPool::SyncRunTask(Task *task, int start_num, int task_num) const {
  // run task sequentially
  // if the current thread is not the actor thread
  float per_scale = kMaxScale / (task_num - start_num);
  for (int i = start_num; i < task_num; ++i) {
    float lhs_scale = i * per_scale;
    float rhs_scale = (i + 1) * per_scale;
    rhs_scale = i == task_num - 1 ? kMaxScale : rhs_scale;
    task->status |= task->func(task->content, i, lhs_scale, rhs_scale);
    (void)++task->finished;
  }
}

void ThreadPool::DistributeTask(Task *task, int task_num, Worker *curr) const {
  int sum_frequency = 0;
  std::vector<Worker *> assigned;
  int num = static_cast<int>(workers_.size()) - 1;
  // if the current thread isn't nullptr, that is the curr is a ActorThread,
  // then assign (task_num - 1) tasks to workers, and run the last one by itself
#ifdef OPERATOR_PARALLELISM
  int num_assigned = task_num;
  bool use_curr = curr != nullptr;
  int count = use_curr ? 1 : 0;
  int offset = static_cast<int>(actor_thread_num_);
#else
  int num_assigned = curr != nullptr ? task_num - 1 : task_num;
  int count = 0;
  int offset = 0;
  bool use_curr = false;

  if (curr != nullptr) {
    use_curr = curr->get_task_free();
  }
#endif
  if (!occupied_actor_thread_) {
    offset = static_cast<int>(actor_thread_num_);
  }

  for (int i = num; i >= offset && count < num_assigned; --i) {
    if (workers_[i]->available()) {
      assigned.push_back(workers_[i]);
      sum_frequency += workers_[i]->frequency();
      (void)++count;
    }
  }

  // when there are not enough free threads,
  // distribute other tasks to the master thread
  if (use_curr) {
#ifdef OPERATOR_PARALLELISM
    assigned.push_back(curr);
    if (count < task_num) {
      auto task_messeages = curr->GetTaskMessages();
      if (task_messeages != nullptr) {
        for (; count < task_num; ++count) {
          task_messeages[count].task = task;
          PushTaskToQueue(&task_messeages[count]);
        }
      }
    }
#else
    for (; count < task_num; ++count) {
      assigned.push_back(curr);
      sum_frequency += curr->frequency();
    }
#endif
  } else if (assigned.size() != static_cast<size_t>(task_num)) {
    CalculateScales(assigned, sum_frequency);
    ActiveWorkers(assigned, task, assigned.size(), curr);
    SyncRunTask(task, assigned.size(), task_num);
    return;
  }

  CalculateScales(assigned, sum_frequency);
  ActiveWorkers(assigned, task, assigned.size(), curr);
}

void ThreadPool::CalculateScales(const std::vector<Worker *> &assigned, int sum_frequency) const {
  // divide task according to computing power(core frequency)
  float lhs_scale = 0;
  float rhs_scale = 0;
  if (sum_frequency == 0) {
    return;
  }
  for (const auto &worker : assigned) {
    THREAD_RETURN_IF_NULL(worker);
    rhs_scale += worker->frequency() * 1.0 / sum_frequency;
    rhs_scale = rhs_scale < 1 ? rhs_scale : 1;
    worker->set_scale(lhs_scale, rhs_scale);
    lhs_scale = rhs_scale;
  }
}

void ThreadPool::ActiveWorkers(const std::vector<Worker *> &workers, Task *task, int task_num,
                               const Worker *curr) const {
  for (int i = 0; i < task_num; ++i) {
    Worker *worker = workers[i];
    THREAD_RETURN_IF_NULL(worker);
    worker->Active(task, i);
    if (worker == curr) {
      (void)worker->RunLocalKernelTask();
    }
  }
}

void ThreadPool::ActiveWorkers() const {
  for (auto &worker : workers_) {
    worker->Active();
  }
}

Worker *ThreadPool::CurrentWorker() const {
  for (const auto &worker : workers_) {
    if (worker->thread_id() == std::this_thread::get_id()) {
      return worker;
    }
  }
  return nullptr;
}

int ThreadPool::InitAffinityInfo() {
#ifdef BIND_CORE
  affinity_ = new (std::nothrow) CoreAffinity();
  THREAD_ERROR_IF_NULL(affinity_);
  int ret = affinity_->InitHardwareCoreInfo();
  if (ret != THREAD_OK) {
    delete affinity_;
    affinity_ = nullptr;
    return THREAD_ERROR;
  }
#endif

#ifdef SERVER_INFERENCE
  server_cpu_frequence = CoreAffinity::GetServerFrequency() / 1000.0f;  // 1GHz = 1000MHz
#endif

  return THREAD_OK;
}

int ThreadPool::SetCpuAffinity(BindMode bind_mode) {
  if (workers_.empty()) {
    return THREAD_ERROR;
  }
  if (affinity_ != nullptr) {
    return affinity_->BindThreads(workers_, bind_mode);
  }
  return THREAD_OK;
}

int ThreadPool::SetCpuAffinity(const std::vector<int> &core_list) {
  if (workers_.empty()) {
    return THREAD_ERROR;
  }
  if (affinity_ != nullptr) {
    return affinity_->BindThreads(workers_, core_list);
  }
  return THREAD_OK;
}

int ThreadPool::SetProcessAffinity(BindMode bind_mode) const {
  if (affinity_ != nullptr) {
    return affinity_->BindProcess(bind_mode);
  }
  return THREAD_OK;
}

void ThreadPool::SetKernelThreadMaxSpinCount(int spin_count) {
  size_t num = workers_.size() - 1;
  for (size_t i = num; i >= actor_thread_num_; i--) {
    THREAD_RETURN_IF_NULL(workers_[i]);
    workers_[i]->SetMaxSpinCount(spin_count);
  }
}

void ThreadPool::SetSpinCountMaxValue() {
  for (auto worker : workers_) {
    THREAD_RETURN_IF_NULL(worker);
    worker->SetMaxSpinCount(max_spin_count_);
  }
  return;
}

void ThreadPool::SetSpinCountMinValue() {
  for (auto worker : workers_) {
    THREAD_RETURN_IF_NULL(worker);
    worker->SetMaxSpinCount(min_spin_count_);
  }
  return;
}

void ThreadPool::SetMaxSpinCount(int spin_count) {
  if (spin_count <= 0) {
    return;
  }
  max_spin_count_ = spin_count;
}

void ThreadPool::SetMinSpinCount(int spin_count) {
  if (spin_count <= 0) {
    return;
  }
  min_spin_count_ = spin_count;
}

ThreadPool *ThreadPool::CreateThreadPool(size_t thread_num, const std::vector<int> &core_list) {
  ThreadPool *pool = new (std::nothrow) ThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret = pool->CreateThreads(thread_num, core_list);
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  return pool;
}

void ThreadPool::SetWorkerIdMap() {
  for (size_t i = 0; i < workers_.size(); ++i) {
    auto thread_id = workers_[i]->thread_id();
    worker_ids_[thread_id] = i;
  }
  return;
}
}  // namespace mindspore
