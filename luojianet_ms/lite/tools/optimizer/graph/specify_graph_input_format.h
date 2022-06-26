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

#ifndef LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_INPUT_FORMAT_H_
#define LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_INPUT_FORMAT_H_

#include "backend/common/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/api/format.h"

namespace luojianet_ms {
namespace opt {
class SpecifyGraphInputFormat : public Pass {
 public:
  explicit SpecifyGraphInputFormat(luojianet_ms::Format format = luojianet_ms::NHWC)
      : Pass("SpecifyGraphInputFormat"), format_(format) {}
  ~SpecifyGraphInputFormat() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS HandleGraphInput(const FuncGraphPtr &graph);
  luojianet_ms::Format format_;
};
}  // namespace opt
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_INPUT_FORMAT_H_
