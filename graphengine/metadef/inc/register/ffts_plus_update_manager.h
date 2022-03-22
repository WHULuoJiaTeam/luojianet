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

#ifndef INC_REGISTER_FFTS_PLUS_UPDATE_MANAGER_H_
#define INC_REGISTER_FFTS_PLUS_UPDATE_MANAGER_H_

#include <map>
#include <string>
#include <memory>
#include <functional>
#include <mutex>

#include "register/ffts_plus_task_update.h"

namespace ge {
using FftsCtxUpdatePtr = std::shared_ptr<FFTSPlusTaskUpdate>;
using FftsCtxUpdateCreatorFun = std::function<FftsCtxUpdatePtr()>;

class FftsPlusUpdateManager {
 public:
  static FftsPlusUpdateManager &Instance();

  /**
   * Get FFTS Plus context by core type.
   * @param core_type: core type of Node
   * @return FFTS Plus Update instance.
   */
  FftsCtxUpdatePtr GetUpdater(const std::string &core_type) const;

  class FftsPlusUpdateRegistrar {
   public:
    FftsPlusUpdateRegistrar(const std::string &core_type, const FftsCtxUpdateCreatorFun &creator) {
      FftsPlusUpdateManager::Instance().RegisterCreator(core_type, creator);
    }
    ~FftsPlusUpdateRegistrar() = default;
  };

 private:
  FftsPlusUpdateManager() = default;
  ~FftsPlusUpdateManager() = default;

  /**
   * Register FFTS Plus context update executor.
   * @param core_type: core type of Node
   * @param creator: FFTS Plus Update instance Creator.
   */
  void RegisterCreator(const std::string &core_type, const FftsCtxUpdateCreatorFun &creator);

  std::mutex mutex_;
  std::map<std::string, FftsCtxUpdateCreatorFun> creators_;
};
} // namespace ge

#define REGISTER_FFTS_PLUS_CTX_UPDATER(core_type, task_clazz)             \
    REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ_HELPER(__COUNTER__, core_type, task_clazz)

#define REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ_HELPER(ctr, type, clazz) \
    REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ(ctr, type, clazz)

#define REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ(ctr, type, clazz)                          \
    ge::FftsPlusUpdateManager::FftsPlusUpdateRegistrar g_##type##_creator##ctr(type, []() { \
      return std::shared_ptr<clazz>(new(std::nothrow) clazz());                             \
    })

#endif // INC_REGISTER_FFTS_PLUS_UPDATE_MANAGER_H_
