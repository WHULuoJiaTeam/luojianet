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

#ifndef INC_EXTERNAL_GE_IR_BUILD_H_
#define INC_EXTERNAL_GE_IR_BUILD_H_

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

#include <string>
#include <map>
#include <memory>
#include "graph/graph.h"
#include "graph/ge_error_codes.h"
namespace ge {
const int32_t IR_MAJOR_VERSION = 1;
const int32_t IR_MINOR_VERSION = 0;
const int32_t IR_PATCH_VERSION = 0;

struct ModelBufferData {
  std::shared_ptr<uint8_t> data = nullptr;
  uint64_t length;
};

enum aclgrphAttrType { ATTR_TYPE_KEEP_DTYPE = 0, ATTR_TYPE_WEIGHT_COMPRESS };

/**
 * @ingroup AscendCL
 * @brief build model.Notice the model is stored in buffer
 *
 * @param global_options[IN] global init params for build
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ATTRIBUTED_DEPRECATED(GE_FUNC_VISIBILITY graphStatus aclgrphBuildInitialize(std::map<AscendString, AscendString> &))
GE_FUNC_VISIBILITY graphStatus aclgrphBuildInitialize(std::map<std::string, std::string> global_options);

GE_FUNC_VISIBILITY graphStatus aclgrphBuildInitialize(std::map<AscendString, AscendString> &global_options);

/**
 * @ingroup AscendCL
 * @brief build model.Notice the model is stored in buffer
 *
 */
GE_FUNC_VISIBILITY void aclgrphBuildFinalize();

/**
 * @ingroup AscendCL
 * @brief build model.Notice the model is stored in buffer
 *
 * @param graph[IN]   the graph ready to build
 * @param options[IN] options used for build
 * @param model[OUT]  builded model
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ATTRIBUTED_DEPRECATED(GE_FUNC_VISIBILITY graphStatus aclgrphBuildModel(const ge::Graph &,
                                                                       const std::map<AscendString, AscendString> &,
                                                                       ModelBufferData &))
GE_FUNC_VISIBILITY graphStatus aclgrphBuildModel(const ge::Graph &graph,
                                                 const std::map<std::string, std::string> &build_options,
                                                 ModelBufferData &model);

GE_FUNC_VISIBILITY graphStatus aclgrphBuildModel(const ge::Graph &graph,
                                                 const std::map<AscendString, AscendString> &build_options,
                                                 ModelBufferData &model);

/**
 * @ingroup AscendCL
 * @brief save model buffer to file
 *
 * @param output_file[IN]   the file path to be saved
 * @param model[IN]         model buffer data
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ATTRIBUTED_DEPRECATED(GE_FUNC_VISIBILITY graphStatus aclgrphSaveModel(const char *, const ModelBufferData &))
GE_FUNC_VISIBILITY graphStatus aclgrphSaveModel(const std::string &output_file, const ModelBufferData &model);

GE_FUNC_VISIBILITY graphStatus aclgrphSaveModel(const char *output_file, const ModelBufferData &model);

/**
 * @ingroup AscendCL
 * @brief query IR interface version
 *
 * @param major_version[OUT] IR interface major version
 * @param minor_version[OUT] IR interface minor version
 * @param patch_version[OUT] IR interface patch version
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
GE_FUNC_VISIBILITY graphStatus aclgrphGetIRVersion(int32_t *major_version, int32_t *minor_version,
                                                   int32_t *patch_version);

/**
 * @ingroup AscendCL
 * @brief dump graph
 *
 * @param graph[IN] the graph ready to build
 * @param file[IN] file path
 * @param file[IN] file path string len
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
GE_FUNC_VISIBILITY graphStatus aclgrphDumpGraph(const ge::Graph &graph, const char *file, const size_t len);

/**
 * @ingroup AscendCL
 * @brief create single op graph
 *
 * @param op_type[IN] the op_type
 * @param inputs[IN] the inputdesc
 * @param outputs[IN] the outputdesc
 * @param graph[OUT] the graph
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
GE_FUNC_VISIBILITY graphStatus aclgrphGenerateForOp(const AscendString &op_type, const std::vector<TensorDesc> &inputs,
                                                    const std::vector<TensorDesc> &outputs, Graph &graph);

/**
 * @name  aclgrphSetOpAttr
 * @brief set attribute for operators in the configuration file
 * @param graph      [IN/OUT] compute graph
 * @param attr_type  [In] attribute type
 * @param cfg_path   [IN] the config file path
 * @return graphStatus
 */
GE_FUNC_VISIBILITY graphStatus aclgrphSetOpAttr(Graph &graph, aclgrphAttrType attr_type, const char *cfg_path);

};      // namespace ge
#endif  // INC_EXTERNAL_GE_IR_BUILD_H_
