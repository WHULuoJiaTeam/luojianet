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

#ifndef INC_FRAMEWORK_OMG_GE_INIT_H_
#define INC_FRAMEWORK_OMG_GE_INIT_H_
#include <map>
#include <string>
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
class GE_FUNC_VISIBILITY GEInit {
 public:
  // GE Environment Initialize, return Status: SUCCESS,FAILED
  static Status Initialize(const std::map<std::string, std::string> &options);

  static std::string GetPath();

  // GE Environment Finalize, return Status: SUCCESS,FAILED
  static Status Finalize();
};
}  // namespace ge

#endif  // INC_FRAMEWORK_OMG_GE_INIT_H_
