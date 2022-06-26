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

#ifndef MINDSPORE_CORE_UTILS_MISC_H_
#define MINDSPORE_CORE_UTILS_MISC_H_

#include <list>
#include <memory>
#include <string>
#include <sstream>

#include "utils/log_adapter.h"

namespace mindspore {
MS_CORE_API extern const int RET_SUCCESS;
MS_CORE_API extern const int RET_FAILED;
MS_CORE_API extern const int RET_CONTINUE;
MS_CORE_API extern const int RET_BREAK;

/// \brief Demangle the name to make it human reablable.
///
/// \param[in] name The name to be demangled.
///
/// \return The demangled name.
MS_CORE_API extern std::string demangle(const char *name);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_MISC_H_
