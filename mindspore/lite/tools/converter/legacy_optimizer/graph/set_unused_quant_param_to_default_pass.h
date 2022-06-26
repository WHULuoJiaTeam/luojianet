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
#ifndef LITE_UNUSED_QUANT_PARAM_DATA_REMOVE_PASS_H
#define LITE_UNUSED_QUANT_PARAM_DATA_REMOVE_PASS_H
#include <memory>
#include "tools/converter/optimizer.h"
#include "tools/converter/converter_flags.h"
#include "tools/common/graph_util.h"
namespace mindspore {
namespace lite {
class SetUnusedQuantParamToDefaultPass : public GraphPass {
 public:
  SetUnusedQuantParamToDefaultPass() {}
  explicit SetUnusedQuantParamToDefaultPass(const converter::Flags &ctx) : ctx_(ctx) {}

  ~SetUnusedQuantParamToDefaultPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  converter::Flags ctx_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // LITE_UNUSED_QUANT_PARAM_DATA_REMOVE_PASS_H
