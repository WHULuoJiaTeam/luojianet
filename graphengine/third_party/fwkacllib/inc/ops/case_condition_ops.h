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
 * \file case_condition_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CASE_CONDITION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CASE_CONDITION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief x[0] is i, x[1] is j and x[2] is k when algorithm is LU,
y = 0 when i >= k && j < k,
y = 1 when i == k && j == k,
y = 2 when i > k && j == k,
y = 3 when i == k && j > k,
y = 4 when i > k && j > k,
default y = 5
use for lu decomposition
*@par Inputs:
*x: A Tensor of type int32/int64/uint64. \n

*@par Attributes:
*algorithm: A string, only support LU now
*@par Outputs:
*y: A Tensor of type int32
*/
REG_OP(CaseCondition)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(algorithm, String, "LU")
    .OP_END_FACTORY_REG(CaseCondition)

} // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_CASE_CONDITION_OPS_H_
