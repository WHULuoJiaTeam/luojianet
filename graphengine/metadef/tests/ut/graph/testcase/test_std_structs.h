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

#ifndef METADEF_CXX_TEST_STD_STRUCTS_H
#define METADEF_CXX_TEST_STD_STRUCTS_H
#include "proto/ge_ir.pb.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"

namespace ge {
GeTensorDesc StandardTd_5d_1_1_224_224();
void ExpectStandardTdProto_5d_1_1_224_224(const proto::TensorDescriptor &input_td);
}
#endif  //METADEF_CXX_TEST_STD_STRUCTS_H
