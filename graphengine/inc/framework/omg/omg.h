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

#ifndef INC_FRAMEWORK_OMG_OMG_H_
#define INC_FRAMEWORK_OMG_OMG_H_

#include <google/protobuf/message.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "proto/ge_ir.pb.h"
#include "proto/om.pb.h"

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "runtime/kernel.h"

using domi::Status;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

namespace ge {
/**
 * @ingroup domi_omg
 * @brief init omg context
 * @return void
 */
GE_FUNC_VISIBILITY Status InitDomiOmgContext(const std::string &input_shape, const std::string &input_format,
                                             const std::string &net_format, bool is_dynamic_input);

/**
 * @ingroup domi_omg
 * @brief generate graph based on the input model file and weight file
 * @param [out] graph graph
 * @param [in] model_file path of model file
 * @param [in] weights_file path of weight file
 * @param [in] type type of the input model
 * @param [in] op_conf op mapping configuration
 * @param [in] target type of platform. If a tiny model is generated, set target to tiny
 * @param [in] run_mode run model
 * @param [in] enable_l2dynamic enable l2dynamic
 * @param [in] is_dynamic_input dynamic input, true of false
 * @param [in] atc_params multiply atc params
 * @return Status result code
 */
GE_FUNC_VISIBILITY Status ParseGraph(ge::Graph &graph, const std::map<std::string, std::string> &atc_params,
                                     const char *model_file, const char *weights_file, domi::FrameworkType type,
                                     const char *op_conf = nullptr, const char *target = nullptr,
                                     RunMode run_mode = RunMode::GEN_OM_MODEL, bool is_dynamic_input = false);

/**
 * @ingroup domi_omg
 * @brief generates a simplified JSON file based on the key value of the offline model file in protobuf format
 * @param [in] model_file path of offline model file
 * @param [out] json_file path of json file
 * @param [key] encrypted key
 * @return Status result code
 */
GE_FUNC_VISIBILITY Status ConvertOm(const char *model_file, const char *json_file, bool is_covert_to_json);

GE_FUNC_VISIBILITY Status ConvertPbtxtToJson(const char *model_file, const char *json_file);
/**
 * @ingroup domi_omg
 * @brief convert the model file in protobuf format into a JSON file.
 * @param [in] framework type of model
 * @param [in] om model_file path of offline model file
 * @param [out] json_file path of json file
 * @param [key] encrypted key
 * @return Status result code
 */
GE_FUNC_VISIBILITY Status ConvertFwkModelToJson(domi::FrameworkType framework, const char *model_file,
                                                const char *json_file);

GE_FUNC_VISIBILITY void GetGroupName(ge::proto::ModelDef &model);

GE_FUNC_VISIBILITY void FindParserSo(const std::string &path, std::vector<std::string> &fileList,
                                     std::string &caffe_parser_path);

GE_FUNC_VISIBILITY Status DumpInfershapeJson(const ge::Graph &graph, const char *json_file);

GE_FUNC_VISIBILITY Status SetOutputNodeInfo(ge::Graph &graph, const std::string &output_type,
                                            const std::string &output_format);

GE_FUNC_VISIBILITY Status GetOutputLeaf(ge::NodePtr node,
                                        std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info);

GE_FUNC_VISIBILITY void CreateOutputNodesInfo(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                                              std::vector<std::string> &output_nodes_name);

GE_FUNC_VISIBILITY void UpdateOmgCtxWithParserCtx();

GE_FUNC_VISIBILITY void UpdateParserCtxWithOmgCtx();

GE_FUNC_VISIBILITY void PrintModelInfo(ge::proto::ModelDef *model_def, uint32_t modeldef_size);
}  // namespace ge

namespace domi {
/**
 * @ingroup domi_omg
 * @brief get omg context
 * @return reference of OmgContext
 */
GE_FUNC_VISIBILITY ge::OmgContext &GetContext();
}  // namespace domi

#endif  // INC_FRAMEWORK_OMG_OMG_H_
