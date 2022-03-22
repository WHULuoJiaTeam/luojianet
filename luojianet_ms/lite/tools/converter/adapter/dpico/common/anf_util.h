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

#ifndef DPICO_COMMON_ANF_UTIL_H_
#define DPICO_COMMON_ANF_UTIL_H_

#include <vector>
#include <string>
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "ir/anf.h"
#include "api/ir/func_graph.h"
#include "ops/primitive_c.h"

using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_NO_CHANGE;
using luojianet_ms::lite::RET_OK;
using luojianet_ms::lite::STATUS;

namespace luojianet_ms {
namespace dpico {
bool CheckPrimitiveType(const luojianet_ms::AnfNodePtr &node, const luojianet_ms::PrimitivePtr &primitive_type);
STATUS GetPrimitiveType(const luojianet_ms::AnfNodePtr &node, std::string *name);
STATUS GetShapeVectorFromParameter(const luojianet_ms::AnfNodePtr &weight, ShapeVector *shape_vector);
std::vector<int> CastToInt(const luojianet_ms::ValuePtr &value);
size_t GetTupleGetItemOutIndex(const luojianet_ms::CNodePtr &tuple_get_item);
STATUS GetOutputShapesFromCNode(const luojianet_ms::CNodePtr &cnode, std::vector<ShapeVector> *output_shapes);
STATUS GetInputShapeFromCNode(const luojianet_ms::CNodePtr &cnode, size_t input_idx, ShapeVector *shape);
STATUS FetchShapeFromAbstract(const luojianet_ms::abstract::AbstractBasePtr &abstract, ShapeVector *shape);
STATUS FetchTypeIdFromAbstract(const luojianet_ms::abstract::AbstractBasePtr &abstract, TypeId *type_id);
int GetAnfNodeOutputShape(const AnfNodePtr &input, ShapeVector *shape_vector);
std::string TypeIdToString(TypeId type_id);
bool CheckInputs(const luojianet_ms::CNodePtr &cnode);
std::string GetCustomOutputName(const AnfNodePtr &node);
luojianet_ms::tensor::TensorPtr CreateTensorInfo(const void *data, size_t data_size, const std::vector<int64_t> &shape,
                                              luojianet_ms::TypeId data_type);
luojianet_ms::AbstractBasePtr CreateTensorAbstract(const std::vector<int64_t> &shape, luojianet_ms::TypeId data_type);
int InitParameterFromTensorInfo(const luojianet_ms::ParameterPtr &param_node,
                                const luojianet_ms::tensor::TensorPtr &tensor_info);
luojianet_ms::abstract::AbstractBasePtr GetCNodeInputAbstract(const luojianet_ms::CNodePtr &cnode, size_t index);
luojianet_ms::abstract::AbstractBasePtr GetAbstractFromAnfNode(const AnfNodePtr &cnode);
luojianet_ms::ParameterPtr BuildIntValueParameterNode(const api::FuncGraphPtr &func_graph, const int32_t &data,
                                                   const std::string &node_name);
luojianet_ms::ParameterPtr BuildIntVecParameterNode(const api::FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                                 const std::string &node_name);
luojianet_ms::ParameterPtr BuildIntVec2DParameterNode(const api::FuncGraphPtr &func_graph,
                                                   const std::vector<std::vector<int32_t>> &data,
                                                   const std::string &node_name);
luojianet_ms::ParameterPtr BuildFloatValueParameterNode(const api::FuncGraphPtr &func_graph, const float &data,
                                                     const std::string &node_name);
luojianet_ms::CNodePtr GenTransposeNode(const api::FuncGraphPtr &func_graph, const luojianet_ms::AnfNodePtr &input_node,
                                     const std::vector<int> &perm, const std::string &cnode_name);
luojianet_ms::tensor::TensorPtr GetTensorInfo(const luojianet_ms::AnfNodePtr &node);
std::vector<std::vector<int>> CastToVec2DInt(const luojianet_ms::ValuePtr &value);
bool GetBoolAttr(const luojianet_ms::AnfNodePtr &node, const std::string &attr_name);
STATUS GetDataTypeAndShape(const luojianet_ms::ParameterPtr &param_node, luojianet_ms::TypeId *data_type,
                           ShapeVector *shape_vector);
STATUS GetShapeVectorFromStringTensor(const luojianet_ms::tensor::TensorPtr &tensor_info, ShapeVector *shape_vector,
                                      size_t *offset);
inline size_t IntToSize(int u) {
  if (u < 0) {
    MS_LOG(WARNING) << "The int value(" << u << ") is less than 0.";
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}
}  // namespace dpico
}  // namespace luojianet_ms

#endif  // DPICO_COMMON_ANF_UTIL_H_
