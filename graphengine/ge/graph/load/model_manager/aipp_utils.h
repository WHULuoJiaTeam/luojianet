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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_AIPP_UTILS_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_AIPP_UTILS_H_

#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/op_desc.h"
#include "proto/insert_op.pb.h"

using std::vector;

namespace ge {
const uint32_t kAippOriginInputIndex = 0;
const uint32_t kAippInfoNum = 6;
const uint32_t kAippInfoFormat = 0;
const uint32_t kAippInfoDataType = 1;
const uint32_t kAippInfoTensorName = 2;
const uint32_t kAippInfoTensorSize = 3;
const uint32_t kAippInfoDimNum = 4;
const uint32_t kAippInfoShape = 5;

class AippUtils {
 public:
  AippUtils() = default;
  ~AippUtils() = default;

  static Status ConvertAippParams2AippInfo(domi::AippOpParams *aipp_params, AippConfigInfo &aipp_info);
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_AIPP_UTILS_H_
