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
 * \file index_to_addr_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_INDEX_TO_ADDR_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INDEX_TO_ADDR_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief get block tensor according to base addr tensor, for hccl remote read to use.
*@par Inputs:
*@li base_addr: A Tensor of type int64/uint64. \n
*@li row:A Tensor of type int64/uint64. \n
*@li col: A Tensor of type int64/uint64.

*@par Outputs:
*addr_table: list of [rank id, host addr, device addr, read size]

*@par Attributes:
*@li ori_shape: An required list int. Shape of base tensor.
*@li block_size: An required list int. Shape of split block tensor.
*@li ori_storage_mode: An optional string from: '"Matrix", "UT"'. Defaults to
"Matrix". Currently only support Matrix storage
*@li block_storage_mode: An optional string from: '"Matrix", "UT"'. Defaults to
"Matrix". Currently only support Matrix storage
*@li rank_id: An optional int of rank id. Defaults is 0
*@li dtype: An optional Type of base tensor. Defaults is DT_FLOAT
*/
REG_OP(IndexToAddr)
    .INPUT(base_addr, TensorType({DT_INT64, DT_UINT64}))
    .INPUT(x, TensorType({DT_INT64, DT_UINT64}))
    .OUTPUT(addrs_table, TensorType({DT_INT64, DT_UINT64}))
    .REQUIRED_ATTR(ori_shape, ListInt)
    .REQUIRED_ATTR(block_size, ListInt)
    .ATTR(ori_storage_mode, String, "Matrix")
    .ATTR(block_storage_mode, String, "Matrix")
    .ATTR(rank_id, Int, 0)
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(IndexToAddr)

} // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_INDEX_TO_ADDR_OPS_H_
