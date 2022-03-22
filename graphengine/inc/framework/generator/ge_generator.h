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

#ifndef INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_
#define INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "external/ge/ge_ir_build.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/op_desc.h"
#include "graph/detail/attributes_holder.h"
#include "framework/omg/omg_inner_types.h"

namespace ge {
class GeRootModel;
class GE_FUNC_VISIBILITY GeGenerator {
 public:
  using InOutTensorRef = std::pair<const std::vector<ge::GeTensor> &, const std::vector<ge::GeTensor> &>;
  static GeGenerator &GetInstance() {
    static GeGenerator Instance;
    return Instance;
  }
  GeGenerator() = default;

  ~GeGenerator() {
    (void)Finalize();
  }

  GeGenerator(const GeGenerator &) = delete;

  GeGenerator &operator=(const GeGenerator &) = delete;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Initialize(const std::map<std::string, std::string> &options, OmgContext &context);

  Status Finalize();

  Status GenerateOfflineModel(const Graph &graph, const std::string &file_name_prefix,
                              const std::vector<GeTensor> &inputs = std::vector<GeTensor>());

  Status GenerateOnlineModel(const Graph &graph, const std::vector<GeTensor> &inputs, ge::ModelBufferData &model);

  Status GenerateInfershapeGraph(const Graph &graph);

  ///
  /// @ingroup ge
  /// @brief: Build single OP in Model.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] model_file_name: name of model file.
  /// @param [in] compile_flag: op build flag, accurate build is 0, fuzz build is 1
  /// @return SUCCESS or FAILED
  ///
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, const std::string &model_file_name,
                            int32_t compile_flag = 0);
  ///
  /// @ingroup ge
  /// @brief: Build single Op into model buff.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] engine_type: engine type.
  /// @param [in] compile_flag: op build flag, accurate build is 0, fuzz build is 1
  /// @param [out] model_buff: model buff of op.
  /// @return SUCCESS or FAILED
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                            ModelBufferData &model_buff);
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, OpEngineType engine_type, int32_t compile_flag,
                            ModelBufferData &model_buff);
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, OpEngineType engine_type, int32_t compile_flag,
                            ModelBufferData &model_buff, GraphStage graph_stage, ComputeGraphPtr &compute_graph);

  ///
  /// @ingroup ge
  /// @brief: Build single Op into model buff.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] graph_name: graph name.
  /// @param [out] graph: graph of single op.
  /// @return SUCCESS or FAILED
  Status BuildSingleOpGraph(OpDescPtr &op_desc, const InOutTensorRef &inputs_outputs, std::string graph_name,
                            Graph &graph, std::vector<std::pair<std::string, std::string>> &inputs_name_type);
  Status BuildOriginalGraphInfo(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                const std::vector<GeTensor> &outputs, const std::string &model_file_name,
                                bool is_offline, int32_t compile_flag, GraphStage graph_stage, Graph &graph,
                                ComputeGraphPtr &compute_graph, bool &fuzz_compile_flag,
                                std::vector<std::pair<std::string, std::string>> &inputs_name_type);

 private:
  Status GenerateModel(const Graph &graph, const std::string &file_name_prefix, const std::vector<GeTensor> &inputs,
                       ge::ModelBufferData &model, bool is_offline = true);
  Status BuildSingleOp(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs,
                       const std::string &model_file_name, OpEngineType engine_type, ModelBufferData &model_buff,
                       ComputeGraphPtr &compute_graph, bool is_offline = true, int32_t compile_flag = 0,
                       GraphStage graph_stage = GraphStage::GRAPH_STAGE_RESERVED);
  bool CheckNoAicore(const ComputeGraphPtr &graph);
  void RemoveConst(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);
  Status CheckForSingleOp(const OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                          const std::vector<GeTensor> &outputs);
  Status InferFormatForSingleOp(const OpDescPtr &op_desc, const Graph &graph);

  using GeRootModelPtr = std::shared_ptr<ge::GeRootModel>;
  Status SetModelNameForDump(const GeRootModelPtr &ge_root_model);
  Status CreateGeneralizedBuildAttrs(const GeRootModelPtr &ge_root_model, const std::vector<GeTensor> &inputs,
                                     const std::vector<GeTensor> &outputs,
                                     const std::vector<std::pair<std::string, std::string>> &inputs_name_type,
                                     std::vector<ge::NamedAttrs> &generalized_build_attrs);

  class Impl;

  std::shared_ptr<Impl> impl_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_
