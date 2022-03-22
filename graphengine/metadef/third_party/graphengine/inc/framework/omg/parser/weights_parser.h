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

#ifndef INC_FRAMEWORK_OMG_PARSER_WEIGHTS_PARSER_H_
#define INC_FRAMEWORK_OMG_PARSER_WEIGHTS_PARSER_H_

#include "graph/graph.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/range_vistor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace domi {
/**
 * @ingroup domi_omg
 * @brief Weight information resolver
 *
 */
class GE_FUNC_VISIBILITY WeightsParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief Constructor
   */
  WeightsParser() {}

  /**
   * @ingroup domi_omg
   * @brief Deconstructor
   */
  virtual ~WeightsParser() {}

  /**
   * @ingroup domi_omg
   * @brief Analyze weight data
   * @param [in] file Path of weight file after training
   * @param [in|out]  graph Graph for saving weight information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  virtual Status Parse(const char *file, ge::Graph &graph) = 0;

  /**
   * @ingroup domi_omg
   * @brief Parse relevant data from memory and save it to graph
   * @param [in] input Model file memory data
   * @param [in|out] graph A graph for saving the model information after analysis
   * @return SUCCESS
   * @return FAILED
   * @author
   */
  virtual Status ParseFromMemory(const char *input, uint32_t lengt, ge::ComputeGraphPtr &graph) = 0;
};
}  // namespace domi

#endif  // INC_FRAMEWORK_OMG_PARSER_WEIGHTS_PARSER_H_
