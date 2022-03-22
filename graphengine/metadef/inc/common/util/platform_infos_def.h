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

#ifndef PLATFORM_INFOS_DEF_H
#define PLATFORM_INFOS_DEF_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "platform_info_def.h"

namespace fe {
class PlatFormInfosImpl;
using PlatFormInfosImplPtr = std::shared_ptr<PlatFormInfosImpl>;
class PlatFormInfos {
 public:
  bool Init();
  std::map<std::string, std::vector<std::string>> GetAICoreIntrinsicDtype();
  std::map<std::string, std::vector<std::string>> GetVectorCoreIntrinsicDtype();
  bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val);

  void SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsic_dtypes);
  void SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsic_dtypes);
  void SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);

 private:
  PlatFormInfosImplPtr platform_infos_impl_{nullptr};
};

class OptionalInfosImpl;
using OptionalInfosImplPtr = std::shared_ptr<OptionalInfosImpl>;
class OptionalInfos {
 public:
  bool Init();
  std::string GetSocVersion();
  std::string GetCoreType();
  uint32_t GetAICoreNum();
  std::string GetL1FusionFlag();

  void SetSocVersion(std::string soc_version);
  void SetCoreType(std::string core_type);
  void SetAICoreNum(uint32_t ai_core_num);
  void SetL1FusionFlag(std::string l1_fusion_flag);
 private:
  OptionalInfosImplPtr optional_infos_impl_{nullptr};
};

}
#endif
