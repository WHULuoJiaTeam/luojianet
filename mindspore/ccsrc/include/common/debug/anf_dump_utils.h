/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ANF_DUMP_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ANF_DUMP_UTILS_H_

#include <string>
#include <memory>
#include <utility>
#include <functional>

#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "include/common/visible.h"

namespace mindspore {
COMMON_EXPORT std::string GetNodeFuncStr(const AnfNodePtr &nd);
COMMON_EXPORT std::string TypeToShortString(const TypeId &typeId);
COMMON_EXPORT std::string GetKernelNodeName(const AnfNodePtr &anf_node);

class COMMON_EXPORT AnfDumpHandler {
  HANDLER_DEFINE(std::string, PrintInputTypeShapeFormat, AnfNodePtr, size_t);
  HANDLER_DEFINE(std::string, PrintOutputTypeShapeFormat, AnfNodePtr, size_t);
  HANDLER_DEFINE(ValuePtr, InStrategyValue, AnfNodePtr);
  HANDLER_DEFINE(ValuePtr, InStrategyStageValue, AnfNodePtr);
  HANDLER_DEFINE(ValuePtr, OutStrategyValue, AnfNodePtr);
  HANDLER_DEFINE(std::string, ValueNodeStr, ValueNodePtr);
  HANDLER_DEFINE(void, DumpDat, std::string, FuncGraphPtr);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ANF_DUMP_UTILS_H_
