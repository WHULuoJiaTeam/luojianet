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

#ifndef INC_GRAPH_UTILS_ANCHOR_UTILS_H_
#define INC_GRAPH_UTILS_ANCHOR_UTILS_H_

#include "graph/anchor.h"
#include "graph/node.h"

namespace ge {
class AnchorUtils {
 public:
  // Get anchor format
  static Format GetFormat(const DataAnchorPtr &data_anchor);

  // Set anchor format
  static graphStatus SetFormat(const DataAnchorPtr &data_anchor, Format data_format);

  // Get anchor status
  static AnchorStatus GetStatus(const DataAnchorPtr &data_anchor);

  // Set anchor status
  static graphStatus SetStatus(const DataAnchorPtr &data_anchor, AnchorStatus anchor_status);

  static bool HasControlEdge(const AnchorPtr &anchor);

  static bool IsControlEdge(const AnchorPtr &src, const AnchorPtr &dst);

  static int32_t GetIdx(const AnchorPtr &anchor);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_ANCHOR_UTILS_H_
