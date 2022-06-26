/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_LOAD_MINDIR_INFER_MINDIR_H
#define MINDSPORE_CORE_LOAD_MINDIR_INFER_MINDIR_H
#include "base/base.h"
#include "ir/anf.h"

namespace mindspore {
MS_CORE_API bool InferMindir(const FuncGraphPtr &root, const AbstractBasePtrList &args, bool raise_exception = false);
MS_CORE_API bool ValidMindir(const FuncGraphPtr &root);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_LOAD_MINDIR_INFER_MINDIR_H
