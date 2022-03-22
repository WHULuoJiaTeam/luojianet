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
 * \file cluster.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CLUSTER_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CLUSTER_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Perform k-means clustering on a data matrix. \n

* @par Inputs:
* Three required inputs and one optional inputs, including: \n
* @li x: A 2D tensor of data type float32. \n
* @li y: A 2D tensor of data type float32. \n
* @li sum_square_x: An optional 2D tensor of data type float32. \n
* @li sum_square_y: A 2D tensor of data type float32. \n

* @par Attributes:
* use_actual_distance: Indicates whether to calculate the complete distance. \n

* @par Outputs:
* @li segment_sum: A tensor of data type float32. \n
* @li segment_count: A tensor of data type float32. \n
* @li k_mean_total_sum: A tensor of data type float32. \n
*/
REG_OP(KMeansCentroids)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .INPUT(sum_square_y, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sum_square_x, TensorType({DT_FLOAT}))
    .OUTPUT(segment_sum, TensorType({DT_FLOAT}))
    .OUTPUT(segment_count, TensorType({DT_FLOAT}))
    .OUTPUT(kmean_total_sum, TensorType({DT_FLOAT}))
    .ATTR(use_actual_distance, Bool, false)
    .OP_END_FACTORY_REG(KMeansCentroids)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_CLUSTER_H_
