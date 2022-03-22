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

#ifndef GE_COMMON_PROPERTIES_MANAGER_H_
#define GE_COMMON_PROPERTIES_MANAGER_H_

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "common/dump/dump_properties.h"
#include "graph/op_desc.h"
#include "framework/common/ge_compiler_options.h"

namespace ge {
// Configuration property management
static const char *SYSMODE GE_ATTRIBUTE_UNUSED = "FMK_SYSMODE";
static const char *USE_FUSION GE_ATTRIBUTE_UNUSED = "FMK_USE_FUSION";
static const char *TIMESTAT_ENABLE GE_ATTRIBUTE_UNUSED = "DAVINCI_TIMESTAT_ENABLE";
static const char *ANNDROID_DEBUG GE_ATTRIBUTE_UNUSED = "ANNDROID_DEBUG";

class PropertiesManager {
 public:
  // Singleton
  static PropertiesManager &Instance();

  /**
   *  @ingroup domi_ome
   *  @brief Initialize configuration parameters, which must be invoked in main.
   *  @param [in] file_path Property profile path
   *  @return true success
   *  @return false fail
   *  @author
   */
  bool Init(const std::string &file_path);

  /**
   *  @ingroup domi_ome
   *  @brief Get configuration parameter value
   *  @param [in] key Configuration parameter name
   *  @return Configuration parameter value. If the parameter name does not exist, return null
   *  @author
   */
  std::string GetPropertyValue(const std::string &key);

  /**
   *  @ingroup domi_ome
   *  @brief Set configuration parameters
   *  @param [in] key Configuration parameter name
   *  @param [out] key Configuration parameter value
   *  @author
   */
  void SetPropertyValue(const std::string &key, const std::string &value);

  /**
   *  @ingroup domi_ome
   *  @brief Return configuration parameters
   *  @return properties_map_
   *  @author
   */
  std::map<std::string, std::string> GetPropertyMap();

  /**
   *  @ingroup domi_ome
   *  @brief Adapt key value pair form, set different separators
   *  @param [in] delimiter
   *  @author
   */
  void SetPropertyDelimiter(const std::string &de);

 private:
  // Private construct, destructor
  PropertiesManager();
  ~PropertiesManager();

  // Get file content
  bool LoadFileContent(const std::string &file_path);

  // Parsing a single line file
  bool ParseLine(const std::string &line);

  // Remove space before and after string
  std::string Trim(const std::string &str);

  bool is_inited_;

  // Configuration item separator, default is "="
  std::string delimiter;

  std::map<std::string, std::string> properties_map_;
  std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_COMMON_PROPERTIES_MANAGER_H_
