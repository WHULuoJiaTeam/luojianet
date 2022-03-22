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


#ifndef COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_
#define COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_
#include <string>
#include <map>

#include "external/graph/types.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"

namespace ge {

class NodeShapeTransUtils {
 public:
  bool Init();
  bool CatchFormatAndShape();
  bool UpdateFormatAndShape();

  explicit NodeShapeTransUtils(const OpDescPtr op_desc) : op_desc_(op_desc), in_num_(0U), out_num_(0U) {
  }

  ~NodeShapeTransUtils() {
  }

 private:
  SmallVector<Format, kDefaultMaxInputNum> map_format_in_;
  SmallVector<Format, kDefaultMaxInputNum> map_ori_format_in_;
  SmallVector<DataType, kDefaultMaxInputNum> map_dtype_in_;
  SmallVector<Format, kDefaultMaxOutputNum> map_format_out_;
  SmallVector<Format, kDefaultMaxOutputNum> map_ori_format_out_;
  SmallVector<DataType, kDefaultMaxOutputNum> map_dtype_out_;

  OpDescPtr op_desc_;
  size_t in_num_;
  size_t out_num_;
};
}  // namespace ge
#endif  // COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_