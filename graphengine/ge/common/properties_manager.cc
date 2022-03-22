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

#include "common/properties_manager.h"

#include <climits>
#include <cstdio>
#include <fstream>

#include "common/ge/ge_util.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"

namespace ge {
PropertiesManager::PropertiesManager() : is_inited_(false), delimiter("=") {}
PropertiesManager::~PropertiesManager() {}

// singleton
PropertiesManager &PropertiesManager::Instance() {
  static PropertiesManager instance;
  return instance;
}

// Initialize property configuration
bool PropertiesManager::Init(const std::string &file_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (is_inited_) {
    GELOGW("Already inited, will be initialized again");
    properties_map_.clear();
    is_inited_ = false;
    return is_inited_;
  }

  if (!LoadFileContent(file_path)) {
    return false;
  }

  is_inited_ = true;
  return is_inited_;
}

// Load file contents
bool PropertiesManager::LoadFileContent(const std::string &file_path) {
  // Normalize the path
  string resolved_file_path = RealPath(file_path.c_str());
  if (resolved_file_path.empty()) {
    DOMI_LOGE("Invalid input file path [%s], make sure that the file path is correct.", file_path.c_str());
    return false;
  }
  std::ifstream fs(resolved_file_path, std::ifstream::in);

  if (!fs.is_open()) {
    GELOGE(PARAM_INVALID, "[Open][File]Failed, file path %s invalid", file_path.c_str());
    REPORT_CALL_ERROR("E19999", "Open file failed, path %s invalid", file_path.c_str());
    return false;
  }

  std::string line;

  while (getline(fs, line)) {  // line not with \n
    if (!ParseLine(line)) {
      GELOGE(PARAM_INVALID, "[Parse][Line]Failed, content is %s", line.c_str());
      REPORT_CALL_ERROR("E19999", "Parse line failed, content is %s", line.c_str());
      fs.close();
      return false;
    }
  }

  fs.close();  // close the file

  GELOGI("LoadFileContent success.");
  return true;
}

// Parsing the command line
bool PropertiesManager::ParseLine(const std::string &line) {
  std::string temp = Trim(line);
  // Comment or newline returns true directly
  if (temp.find_first_of('#') == 0 || *(temp.c_str()) == '\n') {
    return true;
  }

  if (!temp.empty()) {
    std::string::size_type pos = temp.find_first_of(delimiter);
    if (pos == std::string::npos) {
      GELOGE(PARAM_INVALID, "[Check][Param]Incorrect line %s, it must include %s",
             line.c_str(), delimiter.c_str());
      REPORT_CALL_ERROR("E19999", "Incorrect line %s, it must include %s",
                        line.c_str(), delimiter.c_str());
      return false;
    }

    std::string map_key = Trim(temp.substr(0, pos));
    std::string value = Trim(temp.substr(pos + 1));
    if (map_key.empty() || value.empty()) {
      GELOGE(PARAM_INVALID, "[Check][Param]Map_key or value empty, line %s", line.c_str());
      REPORT_CALL_ERROR("E19999", "Map_key or value empty, line %s", line.c_str());
      return false;
    }

    properties_map_[map_key] = value;
  }

  return true;
}

// Remove the space and tab before and after the string
std::string PropertiesManager::Trim(const std::string &str) {
  if (str.empty()) {
    return str;
  }

  std::string::size_type start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return str;
  }

  std::string::size_type end = str.find_last_not_of(" \t\r\n") + 1;
  return str.substr(start, end);
}

// Get property value, if not found, return ""
std::string PropertiesManager::GetPropertyValue(const std::string &map_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = properties_map_.find(map_key);
  if (properties_map_.end() != iter) {
    return iter->second;
  }

  return "";
}

// Set property value
void PropertiesManager::SetPropertyValue(const std::string &map_key, const std::string &value) {
  std::lock_guard<std::mutex> lock(mutex_);
  properties_map_[map_key] = value;
}

// return properties_map_
std::map<std::string, std::string> PropertiesManager::GetPropertyMap() {
  std::lock_guard<std::mutex> lock(mutex_);
  return properties_map_;
}

// Set separator
void PropertiesManager::SetPropertyDelimiter(const std::string &de) {
  std::lock_guard<std::mutex> lock(mutex_);
  delimiter = de;
}

}  // namespace ge
