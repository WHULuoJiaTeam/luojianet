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

#ifndef LUOJIANET_MS_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_PY_H_
#define LUOJIANET_MS_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_PY_H_

#include <memory>

#include "pybind11/pybind11.h"
#include "utils/convert_utils_base.h"
#include "utils/any.h"
#include "base/base_ref.h"
#include "base/base.h"
#include "ir/anf.h"
#include "include/common/visible.h"

namespace py = pybind11;

namespace luojianet_ms {
py::object AnyToPyData(const Any &value);
COMMON_EXPORT py::object BaseRefToPyData(const BaseRef &value);
COMMON_EXPORT py::object BaseRefToPyData(const BaseRef &value, const AbstractBasePtr &output);
COMMON_EXPORT py::object ValueToPyData(const ValuePtr &value);

COMMON_EXPORT bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &output, const py::tuple &args,
                                                     const std::shared_ptr<py::object> &ret_val);
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_PY_H_
