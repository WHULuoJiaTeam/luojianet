/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "backend/common/optimizer/pass_manager.h"

#include <sys/time.h>
#include <deque>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"
#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
void CacheManager::Update(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto type_iter = type_map_.find(node);
  auto shape_iter = shape_map_.find(node);
  if (type_iter != type_map_.end()) {
    (void)type_map_.erase(type_iter);
  }
  if (shape_iter != shape_map_.end()) {
    (void)shape_map_.erase(shape_iter);
  }
}

TypeId CacheManager::GetOutputType(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto iter = type_map_.find(node);
  if (iter != type_map_.end()) {
    auto types = iter->second;
    auto type_iter = types.find(index);
    if (type_iter != types.end()) {
      return type_iter->second;
    }
    return kTypeUnknown;
  }
  auto output_nums = common::AnfAlgo::GetOutputTensorNum(node);
  std::map<size_t, TypeId> index_to_types;
  TypeId result = kTypeUnknown;
  for (size_t i = 0; i < output_nums; i++) {
    auto output_type = common::AnfAlgo::GetOutputInferDataType(node, i);
    (void)index_to_types.emplace(i, output_type);
    if (index == i) {
      result = output_type;
    }
  }
  (void)type_map_.emplace(node, index_to_types);
  return result;
}

std::vector<size_t> CacheManager::GetOutputShape(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto iter = shape_map_.find(node);
  if (iter != shape_map_.end()) {
    auto shapes = iter->second;
    auto shape_iter = shapes.find(index);
    if (shape_iter != shapes.end()) {
      return shape_iter->second;
    }
    return {};
  }
  auto output_nums = common::AnfAlgo::GetOutputTensorNum(node);
  std::map<size_t, std::vector<size_t>> index_to_shapes;
  std::vector<size_t> result = {};
  for (size_t i = 0; i < output_nums; i++) {
    auto output_shape = common::AnfAlgo::GetOutputInferShape(node, i);
    (void)index_to_shapes.emplace(i, output_shape);
    if (index == i) {
      result = output_shape;
    }
  }
  (void)shape_map_.emplace(node, index_to_shapes);
  return result;
}

const std::vector<PassPtr> &PassManager::Passes() const { return passes_; }

void PassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

bool PassManager::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const {
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time {};
  struct timeval end_time {};
  (void)gettimeofday(&start_time, nullptr);
#endif
  bool changed = pass->Run(func_graph);
  constexpr auto kMicroSendUnit = 1000000;
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, kMicroSendUnit>> cost = end_time - start_time;
  MS_LOG(INFO) << "Run pass " + GetPassFullname(pass_id, pass) + " in " << cost.count() << " us";
#else
  (void)gettimeofday(&end_time, nullptr);
  // time unit: us
  uint64_t cost = kMicroSendUnit * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Run pass " + GetPassFullname(pass_id, pass) + " in " << cost << " us";
#endif
  return changed;
}

std::string PassManager::GetPassFullname(size_t pass_id, const PassPtr &pass) const {
  return std::string("hwopt_") + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
}

void PassManager::DumpPassIR(const FuncGraphPtr &func_graph, const std::string &pass_fullname) const {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  static const auto enable_dump = !GetDumpConfig().disable_backend_dump;
  if (save_graphs && enable_dump) {
    std::ostringstream oss;
    oss << "verbose_ir_files"
        << "/";
    oss << (pass_fullname + ".ir");
    DumpIR(oss.str(), func_graph, true);
  }
#endif
}

bool PassManager::Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const {
  if (func_graph == nullptr) {
    return false;
  }
  bool changed = false;
  size_t num = 0;
  for (const auto &pass : passes) {
    if (pass != nullptr) {
      pass->SetCacheManager(cache_manager_);
      changed = RunPass(func_graph, num, pass) || changed;
#ifdef ENABLE_DUMP_IR
      DumpPassIR(func_graph, GetPassFullname(num, pass));
#endif
      num++;
    }
  }
  return changed;
}

bool PassManager::Run(const FuncGraphPtr &func_graph) const {
  bool changed = false;
  // run all passes
  bool change = true;
  while (change) {
    change = Run(func_graph, passes_);
    changed = change || changed;
    if (run_only_once_) {
      break;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
