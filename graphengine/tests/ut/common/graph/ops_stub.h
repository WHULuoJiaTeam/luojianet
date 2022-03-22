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

#ifndef UT_COMMON_GRAPH_OP_STUB_H_
#define UT_COMMON_GRAPH_OP_STUB_H_

#include "external/graph/operator_reg.h"

// for ir
namespace ge {
// Data
REG_OP(Data)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

    // Flatten
    REG_OP(Flatten)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Flatten)

}  // namespace ge

#endif  // UT_COMMON_GRAPH_OP_STUB_H_
