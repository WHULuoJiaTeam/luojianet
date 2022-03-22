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

#include "register/ffts_plus_update_manager.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
FftsPlusUpdateManager &FftsPlusUpdateManager::Instance() {
  static FftsPlusUpdateManager instance;
  return instance;
}

FftsCtxUpdatePtr FftsPlusUpdateManager::GetUpdater(const std::string &core_type) const {
  const auto it = creators_.find(core_type);
  if (it == creators_.end()) {
    GELOGW("Cannot find creator for core type: %s.", core_type.c_str());
    return nullptr;
  }

  return it->second();
}

void FftsPlusUpdateManager::RegisterCreator(const std::string &core_type, const FftsCtxUpdateCreatorFun &creator) {
  if (creator == nullptr) {
    GELOGW("Register null creator for core type: %s", core_type.c_str());
    return;
  }

  std::unique_lock<std::mutex> lk(mutex_);
  const std::map<std::string, FftsCtxUpdateCreatorFun>::const_iterator it = creators_.find(core_type);
  if (it != creators_.cend()) {
    GELOGW("Creator already exist for core type: %s", core_type.c_str());
    return;
  }

  GELOGI("Register creator for core type: %s", core_type.c_str());
  creators_[core_type] = creator;
}
} // namespace ge
