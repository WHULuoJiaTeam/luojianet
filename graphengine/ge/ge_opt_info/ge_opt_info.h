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

#ifndef GE_OPT_INFO_GE_OPT_INFO_H_
#define GE_OPT_INFO_GE_OPT_INFO_H_

#include "ge/ge_api_error_codes.h"
#include "register/register_types.h"

namespace ge {
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY GeOptInfo {
 public:
  GeOptInfo() = default;
  ~GeOptInfo() = default;
  static Status SetOptInfo();
};
}  // namespace ge

#endif  // GE_OPT_INFO_GE_OPT_INFO_H_
