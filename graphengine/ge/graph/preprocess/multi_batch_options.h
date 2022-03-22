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

#ifndef GE_GRAPH_PREPROCESS_MULTI_BATCH_OPTIONS_H_
#define GE_GRAPH_PREPROCESS_MULTI_BATCH_OPTIONS_H_

#include <vector>

#include "external/ge/ge_api_error_codes.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/node.h"

namespace ge {
namespace multibatch {
///
/// @ingroup ge
/// @brief Update Dynamic Param from Options.
/// @param [in] ComputeGraphPtr &graph: the train graph
/// @return SUCCESS: valid / PARAM_INVALID: invalid.
///
Status CheckSequenceOfOptions(ComputeGraphPtr &graph, std::vector<NodePtr> &data_nodes,
                              std::vector<NodePtr> &getnext_nosink_nodes, std::vector<NodePtr> &getnext_sink_nodes);

Status UpdateNameOfInputShape(ComputeGraphPtr &graph, const vector<NodePtr> &data_nodes,
                              const vector<NodePtr> &getnext_nosink_nodes, const vector<NodePtr> &getnext_sink_nodes);

Status DeleteIdentityInsertByAdapter(ComputeGraphPtr &graph);

Status CheckNegativeCountOfOptions(const std::vector<std::vector<int64_t>> &shapes);
///
/// @ingroup ge
/// @brief Init Dynamic Param from Options.
/// @param [out] std::vector<std::vector<int64_t>> &shapes: Result for Params.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
///
bool InitDynamicParams(std::vector<std::vector<int64_t>> &shapes);

///
/// @ingroup ge
/// @brief Check Dynamic Param is invalid.
/// @param [in] const vector<vector<int64_t>> &shapes: Params for check.
/// @return SUCCESS: valid / PARAM_INVALID: invalid.
///
Status CheckDynamicParams(const std::vector<std::vector<int64_t>> &shapes);

///
/// @ingroup ge
/// @brief Get GeShape from configed shape.
/// @param [in] const std::vector<int64_t> &batch_shape: Configed shape.
/// @param [out] GeShape &data_shape: GeShape for configed shape.
/// @return SUCCESS / PARAM_INVALID
///
Status CalcShape(const std::vector<int64_t> &batch_shape, GeShape &data_shape);

///
/// @ingroup ge
/// @brief parse each data's own dynamic dims.
/// @param [in] vector<vector<int64_t>> &shapes: dynamic batch gears info.
/// @param [in] vector<pair<string, vector<int64_t>>> data_name_and_shape: eg:{{data:{1,1,-1,2}}}.
/// @param [out] map<string, vector<vector<int64_t>>> &data_to_dynamic_info: key:data_name. value:dynamic dims.
/// @return SUCCESS / PARAM_INVALID
///
Status ParserDataToDynamicInfo(const vector<vector<int64_t>> &shapes,
                               vector<pair<string, vector<int64_t>>> &data_name_and_shape,
                               map<string, vector<vector<int64_t>>> &data_to_dynamic_info);

///
/// @ingroup ge
/// @brief Set mbatch_dynamic_type on node.
/// @param [in] const OpDescPtr &op_desc: Node for set attribute.
/// @return 0: SUCCESS / others: INTERNAL_ERROR
///
Status StampDynamicType(const OpDescPtr &op_desc);

///
/// @ingroup ge
/// @brief Check dynamic batch Shape.
/// @param [in] const vector<int64_t> &shape: data_shape to be checked.
/// @param [in] const string &data_name: cur data name.
/// @return 0: true/false
///
GE_FUNC_VISIBILITY bool CheckDynamicBatchShape(const vector<int64_t> &shape, const string &data_name);

///
/// @ingroup ge
/// @brief Check Dynamic image size shape.
/// @param [in] unordered_map<string, vector<int64_t>> &shape_map: map of data_name and data_shape.
/// @param [in] const string &data_name: cur data name.
/// @param [in] const std::string &input_format: cur data format.
/// @param [in]  const std::string &input_format: format of input.
/// @return 0: true/false
///
GE_FUNC_VISIBILITY bool CheckDynamicImageSizeShape(const vector<int64_t> &shape, const string &data_name,
                                                   const std::string &input_format);

}  // namespace multibatch
}  // namespace ge
#endif // GE_GRAPH_PREPROCESS_MULTI_BATCH_OPTIONS_H_
