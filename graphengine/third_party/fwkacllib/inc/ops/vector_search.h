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
 * \file vector_search.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
#define OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Generate ADC(asymmetric distance computation) table. \n
*
* @par Inputs:
* Four inputs, including:
* @li query: A Tensor. Must be one of the following types: float16, float32.
* @li code_book: A Tensor. Must be one of the following types: float16, float32.
* @li centroids: A Tensor. Must be one of the following types: float16, float32.
* @li bucket_list: A Tensor. Must be one of the following types: int32, int64.
*
* @par Outputs:
* adc_tables: A Tensor. Must be one of the following types: float16, float32.
*
* @par Attributes:
* distance_type: The string indicates the distance type of ADC tables. Examples: `"l2sqr", "inner_product"`.
The default value is "l2sqr".
*/
REG_OP(GenADC)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(code_book, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(centroids, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(adc_tables, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(distance_type, String, "l2sqr")
    .OP_END_FACTORY_REG(GenADC)

/**
* @brief Finds values and indices of the "k" largest or least elements for the last dimension. \n
*
* @par Inputs:
* Dynamin inputs, including:
* @li actual_count: A Tensor of type int32, the actual number of pq_distance.
* @li pq_distance: A Tensor, Will be updated after calculation. Must be one of the following types: float32, float16. 
* @li grouped_extreme_distance: A Tensor, the extremum in each group. Must be one of the following types: float32, float16.
* @li pq_index: A Tensor of type int32, index corresponding to pq_distance.
* @li pq_ivf: A Tensor of type int32 , the bucket number corresponding to pq_distance.
*
* @par Attributes:
* @li order: A string, indicates the sorting method of topk_pq_distance. \n
* @li k: Int, k maximum or minimum values. \n
* @li group_size: Int, the group size of the extremum. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(TopKPQDistance)
    .DYNAMIC_INPUT(actual_count, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(pq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_INPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_INPUT(pq_ivf, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(pq_index, TensorType({DT_INT32}))
    .OUTPUT(topk_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(topk_ivf, TensorType({DT_INT32}))
    .OUTPUT(topk_index, TensorType({DT_INT32}))
    .ATTR(order, String, "ASC")
    .REQUIRED_ATTR(k, Int)
    .REQUIRED_ATTR(group_size, Int)
    .OP_END_FACTORY_REG(TopKPQDistance)

/**
* @brief Calculate PQ distance. \n
*
* @par Inputs:
* Six inputs, including:
* @li ivf: A Tensor, dtype is uint8.
* @li bucket_list: A Tensor, dtype is int32.
* @li bucket_base_distance: A Tensor, dtype is float16.
* @li bucket_limits: A Tensor, dtype is int32.
* @li bucket_offsets: A Tensor, dtype is int32.
* @li adc_tables: A Tensor. dtype is float16. \n
*
* @par Outputs:
* Five outputs, including:
* @li actual_count: A Tensor, dtype is int32, the first element means the length of processed ivf.
* @li pq_distance: A Tensor, dtype is float16.
* @li grouped_extreme_distance: A Tensor, dtype is float16.
* @li pq_ivf: A Tensor, dtype is int32.
* @li pq_index: A Tensor, dtype is int32. \n
*
* @par Attributes:
* Five attributes, including:
* @li group_size: A Scalar, indicates the group size when compute grouped_extreme_distance.
* @li total_limit: A Scalar, indicates the total length of the outputs.
* @li extreme_mode: A Scalar, indicates the type of extremum, 0 means minimum, and 1 means maximum.
* @li split_count: A Scalar.
* @li split_index: A Scalar. \n
*
*/
REG_OP(ScanPQCodes)
    .INPUT(ivf, TensorType({DT_UINT8}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_base_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_limits, TensorType({DT_INT32}))
    .INPUT(bucket_offsets, TensorType({DT_INT64}))
    .INPUT(adc_tables, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(actual_count, TensorType({DT_INT32}))
    .OUTPUT(pq_distance, TensorType({DT_FLOAT16}))
    .OUTPUT(grouped_extreme_distance, TensorType({DT_FLOAT16}))
    .OUTPUT(pq_ivf, TensorType({DT_INT32}))
    .OUTPUT(pq_index, TensorType({DT_INT32}))
    .REQUIRED_ATTR(total_limit, Int)
    .ATTR(group_size, Int, 64)
    .ATTR(extreme_mode, Int, 0)
    .ATTR(split_count, Int, 1)
    .ATTR(split_index, Int, 0)
    .OP_END_FACTORY_REG(ScanPQCodes)

/**
* @brief Calculate buckets limit and offset. \n

* @par Inputs:
* Three inputs, including:
* @li bucket_list: A 1-D tensor of type int32 with the value of ivf_counts and ivf_offset index. \n
* @li ivf_counts: A 1-D tensor of type int32 with the value of ivf counts. \n
* @li ivf_offset: A 1-D tensor of type int32 or int64 with the value of ivf offset. \n

* @par Attributes:
* total_limit: A int64 type maximum value of the sum of ivf_counts corresponding to bucket_list. \n

* @par Outputs:
* @li buckets_limit: A 1-D tensor of type int32 with the sum <= total_limit. \n
* @li buckets_offset: A 1-D tensor of type int32 or int64 with the value of ivf_offset corresponding to bucket_list. \n
*/
REG_OP(CalcBucketsLimitAndOffset)
    .INPUT(bucket_list, TensorType({DT_INT32}))
    .INPUT(ivf_counts, TensorType({DT_INT32}))
    .INPUT(ivf_offset, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(buckets_limit, TensorType({DT_INT32}))
    .OUTPUT(buckets_offset, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .OP_END_FACTORY_REG(CalcBucketsLimitAndOffset)

/**
* @brief Calculate ProdVirialSeA. \n
*
* @par Inputs:
* Five inputs, including:
* @li net_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li in_deriv: A Tensor. Must be one of the following types: float16, float32, float64.
* @li rij: A Tensor. Must be one of the following types: float16, float32, float64.
* @li nlist: A Tensor. dtype is int32.
* @li natoms: A Tensor. dtype is int32. \n
*
* @par Outputs:
* Two outputs, including:
* @li virial: A Tensor. Must be one of the following types: float16, float32, float64.
* @li atom_virial: A Tensor. Must be one of the following types: float16, float32, float64. \n
*
* @par Attributes:
* Two attributes, including:
* @li n_a_sel: A Scalar.
* @li n_r_sel: A Scalar. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ProdVirialSeA)
    .INPUT(net_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(in_deriv, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(rij, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(nlist, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .OUTPUT(virial, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(atom_virial, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(n_a_sel, Int)
    .REQUIRED_ATTR(n_r_sel, Int)
    .ATTR(nall, Int, 28328)
    .OP_END_FACTORY_REG(ProdVirialSeA)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
