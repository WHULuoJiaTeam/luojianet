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
 * \file slice_write_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SLICE_WRITE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SLICE_WRITE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief write tensor value to tensor x.
*@par Inputs:
*x: A Tensor of type float16/float/double/int32/int64. \n
*begin:A Tensor of type int32/int64. \n
*value: A Tensor of type float16/float/double/int32/int64.
*@par Outputs:
*x: same tensor with input x
*/
REG_OP(SliceWrite)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .INPUT(begin, TensorType({DT_INT32, DT_INT64}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(SliceWrite)

} // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_SLICE_WRITE_OPS_H_
