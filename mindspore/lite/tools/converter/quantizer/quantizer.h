/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_H
#include <unordered_map>
#include <utility>
#include <memory>
#include "schema/inner/model_generated.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "base/base.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/quant_param_holder.h"

namespace mindspore::lite::quant {
class Quantizer {
 public:
  explicit Quantizer(const converter::Flags &config) : flags_(config) {}

  virtual ~Quantizer() = default;

  virtual int DoQuantize(FuncGraphPtr func_graph) = 0;

 protected:
  converter::Flags flags_;
};
}  // namespace mindspore::lite::quant
#endif
