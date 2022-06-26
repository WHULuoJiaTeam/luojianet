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

#ifndef MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_GRAPH_BUILDER_H_
#define MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_GRAPH_BUILDER_H_

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <utility>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/convert.h"
#include "include/common/visible.h"

namespace mindspore {
namespace transform {
COMMON_EXPORT Status BuildDatasetGraph(const DatasetGraphParam &param, const std::string &phase = "dataset");
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_GRAPH_BUILDER_H_
