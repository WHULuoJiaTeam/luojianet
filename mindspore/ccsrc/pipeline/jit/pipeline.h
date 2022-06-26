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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <unordered_map>
#include <list>

#include "pybind11/pybind11.h"

#include "ir/anf.h"
#include "ir/tensor.h"
#include "pipeline/jit/action.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/graph_compiler/transform.h"
#include "pipeline/jit/base.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
extern const char kMsConvert[];
extern const char kMsVm[];

// namespace to support pipeline structures definition
namespace pipeline {

namespace py = pybind11;

class Pipeline {
 public:
  Pipeline(const ResourcePtr &res, const std::vector<ActionItem> &actions) : resource_(res), actions_(actions) {}

  ~Pipeline() = default;

  void Run();

  ResourcePtr resource() { return resource_; }

  bool NeedCreateBackend();

 private:
  ResourcePtr resource_;
  std::vector<ActionItem> actions_;
};

// A function pipeline.
class GraphExecutorPy : public std::enable_shared_from_this<GraphExecutorPy> {
 public:
  static std::shared_ptr<GraphExecutorPy> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<GraphExecutorPy>(new (std::nothrow) GraphExecutorPy());
    }
    return executor_;
  }

  ~GraphExecutorPy();

  const std::string &phase() const { return phase_; }
  const std::map<std::string, std::string> &jit_config() const { return jit_config_; }
  void SaveCompiledGraph(const std::string &phase);
  bool CompileInner(const py::object &source_obj, const py::tuple &args, const py::object &phase_obj, bool use_vm);
  bool Compile(const py::object &source_obj, const py::tuple &args, const py::object &phase_obj, bool use_vm);

  void ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *arg_list);

  // for pynative mode when use_vm is on
  py::object Run(const py::tuple &args, const py::object &phase_obj);
  ResourcePtr GetResource(const std::string &phase);
  FuncGraphPtr GetFuncGraph(const std::string &phase);
  FuncGraphPtr GetGradGraph(const std::string &phase);
  void SetGradGraph(const FuncGraphPtr &grad_graph, const std::string &phase);
  py::bytes GetFuncGraphProto(const std::string &phase, const std::string &type);
#ifndef ENABLE_SECURITY
  py::bytes GetOptimizeGraphProto(const std::string &phase);
#endif
  void SetJitConfig(const py::dict &jit_config);
  compile::VmEvalFuncPtr GetVmEvalFunc(const std::string &phase);
  bool HasCompiled(const std::string &phase) const;

  FuncGraphPtr BuildGraph(const py::dict &init_params, const std::string &phase,
                          const py::object &broadcast_params = {}) const;
  void UpdataParamNodeDefaultInput(const std::string &phase,
                                   const std::unordered_map<std::string, tensor::TensorPtr> &params);
  void RunInitGraph(const py::dict &init_params, const std::string &phase) const;
  void PyExePath(const py::object &py_exe_path);
  void KernelBuildServerDir(const py::object &kernel_build_server_dir);
  py::dict GetParameterLayout(const std::string &phase);
  // Get CNode name, input node name and attribute from each graph
  py::dict GetParallelGraphInfo(const std::string &phase);
  py::dict GetCNodeStrategy(const std::string &phase);
  py::list GetParallelParameterNameList(const std::string &phase);
  void SetCNodeStrategy(const std::string &name, const parallel::Strategys &strategy);
  size_t GetNumOpsInfo(const std::string &phase);
  void SetNumOpsInfo(size_t);
  py::dict GetAllreduceFusion(const std::string &phase);
  void DelNetRes(const py::set &id);
  void ReleaseResource(const py::object &phase_obj);
  static void ClearRes();
  void set_queue_name(const std::string &queue_name) { queue_name_ = queue_name; }
  void set_enable_tuple_broaden(bool enable_tuple_broaden) { enable_tuple_broaden_ = enable_tuple_broaden; }
  void set_compile_cache_dep_files(const py::list &compile_cache_dep_files) {
    compile_cache_dep_files_ = compile_cache_dep_files;
  }
  void set_weights_values(const py::dict &weights) { weights_ = weights; }
#ifdef ENABLE_DEBUGGER
  void TerminateDebugger();
#endif

  std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> FetchInfoForQuantExport(
    const std::string &phase);

  // Generate a key for mapping function graph
  py::object GenerateArgumentsKey(const py::tuple &args, bool enable_tuple_broaden = false);

 private:
  GraphExecutorPy() = default;
  void GetWeightInfo(const CNodePtr &root_node, const AnfNodePtr &weight_node,
                     std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> *fake_quant_table);
  void GetGeBackendPolicy() const;
  // filter some pipeline actions according to phase, e.g. when exporting onnx, it is no need to execute actions after
  // 'validate' stage
  static std::vector<ActionItem> FilterActions(const std::vector<ActionItem> &actions, const std::string &phase);

  void DelOneNetRes(const py::handle &py_phase);
  // If enable compile cache, get the compile cache resource.
  void InitCompileCacheInfo(const ResourcePtr &resource, const std::string &phase);

  std::map<std::string, ExecutorInfoPtr> info_;
  static std::shared_ptr<GraphExecutorPy> executor_;
  static std::mutex instance_lock_;
  std::map<std::string, py::dict> stra_dict_;
  std::string phase_ = "";
  std::map<std::string, std::string> jit_config_;
  std::map<std::string, size_t> phase_to_num_op_info_;
  std::string queue_name_;
  bool enable_tuple_broaden_{false};
  py::list compile_cache_dep_files_;
  bool compile_cache_consistent_{true};
  py::dict weights_;
  std::map<PyObject *, AbstractBasePtr> cur_convert_input_;
};
using GraphExecutorPyPtr = std::shared_ptr<GraphExecutorPy>;

std::string GetJitLevel();

void CheckArgsValid(const py::object &source_obj, const py::tuple &args);
py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs);

bool InitDistribute(const std::map<std::string, std::string> &options);

void ResetOpId();
void InitHccl();
void FinalizeHccl();
uint32_t GetHcclRankId();
uint32_t GetHcclRankSize();
void InitPipeline();
void FinalizeBackend();
void ClearResAtexit();
void ReleaseGeTsd();
void MemoryRecycle();

void ExportGraph(const std::string &file_name, const std::string &, const std::string &phase);
FuncGraphPtr LoadMindIR(const std::string &file_name, char *dec_key, const size_t key_len, const std::string &dec_mode);

// init and exec dataset sub graph
bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &phase, bool need_run);

// Build and run dataset subgraph for ms backend
bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run);

void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list);

py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len, const std::string &enc_mode);
py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len, const std::string &dec_mode);
bool PyIsCipherFile(const std::string &file_path);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_
