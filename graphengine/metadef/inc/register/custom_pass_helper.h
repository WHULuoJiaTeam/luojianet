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

#ifndef INC_REGISTER_CUSTOM_PASS_HELPER_H_
#define INC_REGISTER_CUSTOM_PASS_HELPER_H_

#include <set>
#include "external/ge/ge_api_error_codes.h"
#include "external/register/register_pass.h"
#include "external/register/register_types.h"

namespace ge {
class CustomPassGreater : std::greater<PassRegistrationData> {
 public:
  bool operator()(const PassRegistrationData &a, const PassRegistrationData &b) const {
    return a.GetPriority() < b.GetPriority();
  }
};

class CustomPassHelper {
 public:
  static CustomPassHelper &Instance();

  void Insert(const PassRegistrationData &);

  Status Run(ge::GraphPtr &);

  ~CustomPassHelper() = default;

 private:
  CustomPassHelper() = default;
  std::multiset<PassRegistrationData, CustomPassGreater> registration_datas_;
};
} // namespace ge

#endif // INC_REGISTER_CUSTOM_PASS_HELPER_H_
