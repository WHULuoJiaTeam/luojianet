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

#ifndef PLATFORM_INFO_H
#define PLATFORM_INFO_H

#include <map>
#include <string>
#include "platform_info_def.h"
#include "platform_infos_def.h"

namespace fe {
class PlatformInfoManager {
 public:
  PlatformInfoManager(const PlatformInfoManager &) = delete;
  PlatformInfoManager &operator=(const PlatformInfoManager &) = delete;

  static PlatformInfoManager &Instance();
  uint32_t InitializePlatformInfo();
  uint32_t Finalize();

  uint32_t GetPlatformInfo(const std::string SoCVersion,
                           PlatformInfo &platform_info,
                           OptionalInfo &opti_compilation_info);

  uint32_t GetPlatformInfoWithOutSocVersion(PlatformInfo &platform_info,
                                            OptionalInfo &opti_compilation_info);

  void SetOptionalCompilationInfo(OptionalInfo &opti_compilation_info);

  uint32_t GetPlatformInfos(const std::string SoCVersion,
                            PlatFormInfos &platform_info,
                            OptionalInfos &opti_compilation_info);

  uint32_t GetPlatformInfoWithOutSocVersion(PlatFormInfos &platform_info,
                                            OptionalInfos &opti_compilation_info);

  void SetOptionalCompilationInfo(OptionalInfos &opti_compilation_info);

 private:
  PlatformInfoManager();
  ~PlatformInfoManager();

  uint32_t LoadIniFile(std::string ini_file_real_path);

  void Trim(std::string &str);

  uint32_t LoadConfigFile(std::string real_path);

  std::string RealPath(const std::string &path);

  std::string GetSoFilePath();

  void ParseVersion(std::map<std::string, std::string> &version_map,
                    std::string &soc_version,
                    PlatformInfo &platform_info_temp);

  void ParseSocInfo(std::map<std::string, std::string> &soc_info_map,
                    PlatformInfo &platform_info_temp);

  void ParseCubeOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                             PlatformInfo &platform_info_temp);

  void ParseBufferOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                               PlatformInfo &platform_info_temp);

  void ParseUBOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                           PlatformInfo &platform_info_temp);

  void ParseUnzipOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                              PlatformInfo &platform_info_temp);

  void ParseAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                       PlatformInfo &platform_info_temp);

  void ParseBufferOfAICoreMemoryRates(std::map<std::string, std::string> &ai_core_memory_rates_map,
                                      PlatformInfo &platform_info_temp);

  void ParseAICoreMemoryRates(std::map<std::string, std::string> &ai_core_memory_rates_map,
                              PlatformInfo &platform_info_temp);

  void ParseUBOfAICoreMemoryRates(std::map<std::string, std::string> &ai_core_memory_rates_map,
                                  PlatformInfo &platform_info_temp);

  void ParseAICoreintrinsicDtypeMap(std::map<std::string, std::string> &ai_coreintrinsic_dtype_map,
                                    PlatformInfo &platform_info_temp);

  void ParseVectorCoreSpec(std::map<std::string, std::string> &vector_core_spec_map,
                           PlatformInfo &platform_info_temp);

  void ParseVectorCoreMemoryRates(std::map<std::string, std::string> &vector_core_memory_rates_map,
                                  PlatformInfo &platform_info_temp);

  void ParseCPUCache(std::map<std::string, std::string> &CPUCacheMap,
                     PlatformInfo &platform_info_temp);

  void ParseVectorCoreintrinsicDtypeMap(std::map<std::string, std::string> &vector_coreintrinsic_dtype_map,
                                        PlatformInfo &platform_info_temp);

  uint32_t ParsePlatformInfoFromStrToStruct(std::map<std::string, std::map<std::string, std::string>> &content_info_map,
                                            std::string &soc_version,
                                            PlatformInfo &platform_info_temp);

  void ParseAICoreintrinsicDtypeMap(std::map<std::string, std::string> &ai_coreintrinsic_dtype_map,
                                    PlatFormInfos &platform_info_temp);

  void ParseVectorCoreintrinsicDtypeMap(std::map<std::string, std::string> &vector_coreintrinsic_dtype_map,
                                        PlatFormInfos &platform_info_temp);

  void ParsePlatformRes(const std::string &label,
                        std::map<std::string, std::string> &platform_res_map,
                        PlatFormInfos &platform_info_temp);

  uint32_t ParsePlatformInfo(std::map<std::string, std::map<std::string, std::string>> &content_info_map,
                             std::string &soc_version,
                             PlatFormInfos &platform_info_temp);

  uint32_t AssemblePlatformInfoVector(std::map<std::string, std::map<std::string, std::string>> &content_info_map);

 private:
  bool init_flag_;
  std::map<std::string, PlatformInfo> platform_info_map_;

  OptionalInfo opti_compilation_info_;

  std::map<std::string, PlatFormInfos> platform_infos_map_;

  OptionalInfos opti_compilation_infos_;
};
}  // namespace fe
#endif
