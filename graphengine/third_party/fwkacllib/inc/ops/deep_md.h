/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

/*!
 * \file deep_md.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_DEEP_MD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_DEEP_MD_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Calculate ProdForceSeA. \n
*
* @par Inputs:
* Five inputs, including:
* @li net_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li in_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li nlist: A Tensor. dtype is int32.
* @li natoms: A Tensor. dtype is int32. \n
*
* @par Outputs:
* atom_force: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Attributes:
* Two attributes, including:
* @li n_a_sel: A Scalar.
* @li n_r_sel: A Scalar. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ProdForceSeA)
    .INPUT(net_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(in_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(nlist, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .OUTPUT(atom_force, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(n_a_sel, Int)
    .REQUIRED_ATTR(n_r_sel, Int)
    .OP_END_FACTORY_REG(ProdForceSeA)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_DEEP_MD_H_
