/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file coordinates_1d_to_2d_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_COORDINATES_1D_TO_2D_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_COORDINATES_1D_TO_2D_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Convert one-dimensional coordinates to two-dimensional coordinates.
*@par Inputs:
*@li x: A Tensor of type int32/int64/uint64. One-dimensional coordinates.
*@li shape: A Tensor of type int32/int64/uint64. 4D tensor [N,C,H,W].
*@par Outputs:
*@li row: row of two-dimensional
*@li col: col of two-dimensional
*@li n: col number of two-dimensional
*/
REG_OP(Coordinates1DTo2D)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(row, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(col, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(n, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OP_END_FACTORY_REG(Coordinates1DTo2D)

} // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_COORDINATES_1D_TO_2D_OPS_H_
