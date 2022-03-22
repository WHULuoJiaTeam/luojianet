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

#ifndef GE_GRAPH_OPTIMIZE_COMMON_PARAMS_H_
#define GE_GRAPH_OPTIMIZE_COMMON_PARAMS_H_

#include <string>

#include "common/singleton.h"
#include "common/types.h"

namespace ge {
class Params : public Singleton<Params> {
 public:
  DECLARE_SINGLETON_CLASS(Params);

  void SetTarget(const char* target) {
    std::string tmp_target = (target != nullptr) ? target : "";

#if defined(__ANDROID__) || defined(ANDROID)
    target_ = "LITE";
    target_8bit_ = TARGET_TYPE_LTTE_8BIT;
#else
    target_ = "MINI";
    target_8bit_ = TARGET_TYPE_MINI_8BIT;
#endif
    if (tmp_target == "mini") {
      target_ = "MINI";
      target_8bit_ = TARGET_TYPE_MINI_8BIT;
    } else if (tmp_target == "lite") {
      target_ = "LITE";
      target_8bit_ = TARGET_TYPE_LTTE_8BIT;
    }
  }

  string GetTarget() const { return target_; }

  uint8_t GetTarget_8bit() const { return target_8bit_; }
  ~Params() override = default;

 private:
  Params() : target_("MINI") {}

  string target_;
  uint8_t target_8bit_ = 0;
};
}  // namespace ge

#endif  // GE_GRAPH_OPTIMIZE_COMMON_PARAMS_H_
