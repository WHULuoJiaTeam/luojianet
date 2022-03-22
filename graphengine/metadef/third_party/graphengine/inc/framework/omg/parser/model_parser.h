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

#ifndef INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_
#define INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_

#include <google/protobuf/message.h>
#include "framework/omg/parser/parser_types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/range_vistor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

using Status = domi::Status;

namespace domi {
using GetGraphCallback = std::function<std::unique_ptr<google::protobuf::Message>(
  const google::protobuf::Message *root_proto, const std::string &graph)>;

using GetGraphCallbackV2 = std::function<std::string(const std::string &subgraph_name)>;

class GE_FUNC_VISIBILITY ModelParser {
 public:
  ModelParser() {}

  virtual ~ModelParser() {}

  /**
   * @ingroup domi_omg
   * @brief Analyze network model data
   * @param [in] file  Network model file path
   * @param [in|out]  graph Save the network information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status Parse(const char *file, ge::Graph &graph) = 0;

  /**
   * @ingroup domi_omg
   * @brief Parse relevant data from memory and save it to graph
   * @param [in] input Model file memory data
   * @param [in] input Model file memory size
   * @param [in|out] graph A graph for saving the model information after analysis
   * @return SUCCESS
   * @return FAILED
   * @author
   */
  virtual Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) = 0;

  /**
   * @ingroup domi_omg
   * @brief Parse relevant data from memory and save it to graph
   * @param [in] input Model file memory data
   * @param [in] input Model file memory size
   * @param [in|out] graph A graph for saving the model information after analysis
   * @return SUCCESS
   * @return FAILED
   * @author
   */
  virtual Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph) = 0;

  /**
   * @ingroup domi_omg
   * @brief Analyze network model data
   * @param [in] proto  network model
   * @param [in|out]  graph Save the network information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) = 0;

  /**
   * @ingroup domi_omg
   * @brief Analyze callback model data in subgraph
   * @param [in] proto network model
   * @param [in] callback callback of subgraph
   * @param [in|out] graph Save the network information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status ParseProtoWithSubgraph(const google::protobuf::Message *proto, GetGraphCallback callback,
                                        ge::ComputeGraphPtr &graph) = 0;
  /**
   * @ingroup domi_omg
   * @brief Convert model files to JSON format
   * @param [in] model_file  Model file path to be converted
   * @param [out] json_file Converted JSON file path
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status ToJson(const char *model_file, const char *json_file) { return domi::SUCCESS; }

  /*
   * @ingroup domi_omg
   * @brief Convert network data type
   * @param [in] type Data type to be converted
   * @return ge::DataType
   */
  virtual ge::DataType ConvertToGeDataType(const uint32_t type) = 0;

  virtual Status ParseAllGraph(const google::protobuf::Message *root_proto, ge::ComputeGraphPtr &root_graph) = 0;

  /**
   * @ingroup domi_omg
   * @brief Analyze network model data
   * @param [in] proto  serialized network model
   * @param [in|out]  graph Save the network information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status ParseProto(const std::string &serialized_proto, ge::ComputeGraphPtr &graph) { return UNSUPPORTED; }

  /**
   * @ingroup domi_omg
   * @brief Analyze callback model data in subgraph
   * @param [in] proto serialized network model
   * @param [in] callback callback of subgraph
   * @param [in|out] graph Save the network information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status ParseProtoWithSubgraph(const std::string &serialized_proto, GetGraphCallbackV2 callback,
                                        ge::ComputeGraphPtr &graph) {
    return UNSUPPORTED;
  }
};
}  // namespace domi

#endif  // INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_
