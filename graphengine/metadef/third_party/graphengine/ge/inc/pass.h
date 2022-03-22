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

#ifndef GE_INC_PASS_H_
#define GE_INC_PASS_H_

#include <memory>

#include "common/fmk_error_codes.h"

namespace ge {
///
/// @ingroup domi_omg
/// @brief pass
/// @author
///
template <typename T>
class Pass {
 public:
  virtual ~Pass() {}
  ///
  /// run pass
  /// @author
  ///
  virtual Status Run(std::shared_ptr<T>) = 0;
};
}  // namespace ge

#endif  // GE_INC_PASS_H_
