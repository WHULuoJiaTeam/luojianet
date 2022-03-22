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
#ifndef GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_
#define GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <mutex>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "external/graph/operator.h"
#include "external/../register/register.h"

namespace ge {
class GE_FUNC_VISIBILITY HostCpuEngine {
 public:
  ~HostCpuEngine() = default;

  static HostCpuEngine &GetInstance() {
    static HostCpuEngine instance;
    return instance;
  }

  ge::Status Initialize();

  void Finalize();

  static bool CheckSupported(const string &op_type);

  ge::Status Run(NodePtr &node, const vector<ConstGeTensorPtr> &inputs, std::vector<GeTensorPtr> &outputs);

  void *GetConstantFoldingHandle() const { return constant_folding_handle_; }

 private:
  HostCpuEngine() = default;

  void CloseSo();

  ge::Status LoadLibs(std::vector<std::string> &lib_paths);

  ge::Status LoadLib(const std::string &lib_path);

  static ge::Status GetRealPath(std::string &path);

  static ge::Status GetLibPath(std::string &lib_path);

  static ge::Status ListSoFiles(const std::string &base_dir, std::vector<std::string> &names);

  static bool IsSoFile(const std::string &file_name);

  static ge::Status FindOpKernel(const NodePtr &node, std::unique_ptr<HostCpuOp> &op_kernel);

  static ge::Status PrepareInputs(const ConstOpDescPtr &op_desc, const vector<ConstGeTensorPtr> &inputs,
                                  std::map<std::string, const Tensor> &named_inputs);

  static ge::Status PrepareOutputs(const ConstOpDescPtr &op_desc, vector<GeTensorPtr> &outputs,
                                   std::map<std::string, Tensor> &named_outputs);

  static ge::Status RunInternal(const OpDescPtr &op_desc, HostCpuOp &op_kernel,
                                std::map<std::string, const Tensor> &named_inputs,
                                std::map<std::string, Tensor> &named_outputs);

  std::mutex mu_;
  std::vector<void *> lib_handles_;
  void *constant_folding_handle_ = nullptr;
  bool initialized_ = false;
};
}  // namespace ge
#endif  // GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_
