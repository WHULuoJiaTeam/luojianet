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

#ifndef GE_GRAPH_COMMON_TRANSOP_UTIL_H_
#define GE_GRAPH_COMMON_TRANSOP_UTIL_H_

#include <string>
#include <unordered_map>

#include "graph/node.h"

namespace ge {
class TransOpUtil {
 public:
  static bool IsTransOp(const NodePtr &node);

  static bool IsTransOp(const std::string &type);

  static int GetTransOpDataIndex(const NodePtr &node);

  static int GetTransOpDataIndex(const std::string &type);

  static bool CheckPrecisionLoss(const NodePtr &src_node);

  static std::string TransopMapToString();

 private:
  TransOpUtil();

  ~TransOpUtil();

  static TransOpUtil &Instance();

  typedef std::map<std::string, int> transop_index_op;
  transop_index_op transop_index_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_COMMON_TRANSOP_UTIL_H_
