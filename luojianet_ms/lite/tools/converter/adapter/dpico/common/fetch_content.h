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

#ifndef LUOJIANET_MS_LITE_TOOLS_ANF_EXPORTER_FETCH_CONTENT_H_
#define LUOJIANET_MS_LITE_TOOLS_ANF_EXPORTER_FETCH_CONTENT_H_

#include <string>
#include <vector>
#include "ir/primitive.h"
#include "ir/func_graph.h"

namespace luojianet_ms {
namespace dpico {
struct DataInfo {
  int data_type_;
  std::vector<int> shape_;
  std::vector<uint8_t> data_;
  DataInfo() : data_type_(0) {}
};

int FetchFromDefaultParam(const ParameterPtr &param_node, DataInfo *data_info);

int FetchDataFromParameterNode(const CNodePtr &cnode, size_t index, DataInfo *data_info);
int GetDataSizeFromTensor(DataInfo *data_info, int *data_size);
}  // namespace dpico
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_LITE_TOOLS_ANF_EXPORTER_FETCH_CONTENT_H_
