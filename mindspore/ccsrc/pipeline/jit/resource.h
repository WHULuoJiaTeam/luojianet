/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_RESOURCE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_RESOURCE_H_

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "utils/any.h"
#include "utils/profile.h"
#include "ir/manager.h"

#include "pipeline/jit/resource_base.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "load_mindir/load_model.h"
#include "pipeline/jit/compile_cache_manager.h"

namespace mindspore {
namespace pipeline {

namespace py = pybind11;

const char kBackend[] = "backend";
const char kStepParallelGraph[] = "step_parallel";
const char kOutput[] = "output";
const char kPynativeGraphId[] = "graph_id";

class InferenceResource;

using BuiltInTypeMap = mindspore::HashMap<int64_t, mindspore::HashMap<std::string, Any>>;

BuiltInTypeMap &GetMethodMap();

BuiltInTypeMap &GetAttrMap();

class Resource : public ResourceBase {
 public:
  explicit Resource(const py::object &obj = py::none());

  ~Resource() override;

  abstract::AnalysisEnginePtr engine() { return engine_; }

  static bool IsTypeInBuiltInMap(const TypeId &type);

  static Any GetMethodPtr(const TypeId &type, const std::string &name);

  static Any GetAttrPtr(const TypeId &type, const std::string &name);

  const py::object &source_input() const { return source_input_; }

  FuncGraphPtr func_graph() const { return func_graph_; }
  void set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = func_graph; }

  FuncGraphPtr optimize_graph() const { return optimize_graph_; }
  void set_optimize_graph(const FuncGraphPtr &optimize_graph) { optimize_graph_ = optimize_graph; }

  const abstract::AbstractBasePtrList &args_spec() const { return args_spec_; }
  void set_args_spec(const abstract::AbstractBasePtrList &args_spec) { args_spec_ = args_spec; }

  void set_vm_loop(const bool &flag, const int64_t size) {
    vm_loop_flag_ = flag;
    loop_size_ = size;
  }
  void set_is_load(bool flag) { is_load_ = flag; }
  bool is_load() const { return is_load_; }
  bool vm_loop_flag() const { return vm_loop_flag_; }
  int64_t loop_size() const { return loop_size_; }

  const LayoutMap &layout_map() const { return layout_map_; }

  // Get the cached func_graph and parameters layout map.
  void GetCompileCacheResource(const py::list &compile_cache_dep_files, const py::dict &weights,
                               const std::string &queue_name, size_t compile_cache_id, bool *compile_cache_consistent);
  void CacheFuncGraph() const;
  bool EnableCompileCache() const { return compile_cache_manager_ != nullptr; }

  // Reclaim resource and clear the cache.
  // GraphExecutorPy::Compile() can be called multiple times, so cache
  // should be cleared.
  void Clean();

 private:
  abstract::AnalysisEnginePtr engine_;
  FuncGraphPtr func_graph_;
  FuncGraphPtr optimize_graph_;
  abstract::AbstractBasePtrList args_spec_;
  // The source obj to compile, usually a `Cell` or `ms_function` decorated function.
  py::object source_input_;
  bool is_cleaned_;
  // The func_graph_ is loaded from mindir
  bool is_load_{false};
  bool vm_loop_flag_{false};
  int64_t loop_size_{1};
  LayoutMap layout_map_{};
  CompileCacheManagerPtr compile_cache_manager_{nullptr};
};

using ResourcePtr = std::shared_ptr<pipeline::Resource>;

}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_RESOURCE_H_
