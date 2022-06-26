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

#ifndef LUOJIANET_MS_CORE_UTILS_FILE_UTILS_H_
#define LUOJIANET_MS_CORE_UTILS_FILE_UTILS_H_

#include <string>
#include <optional>
#include "utils/visible.h"

namespace luojianet_ms {
class MS_CORE_API FileUtils {
 public:
  FileUtils() = default;
  ~FileUtils() = default;

  static std::optional<std::string> GetRealPath(const char *path);
  static void SplitDirAndFileName(const std::string &path, std::optional<std::string> *prefix_path,
                                  std::optional<std::string> *file_name);
  static void ConcatDirAndFileName(const std::optional<std::string> *dir, const std::optional<std::string> *file_name,
                                   std::optional<std::string> *path);
  static std::optional<std::string> CreateNotExistDirs(const std::string &path,
                                                       const bool support_relative_path = false);
#if defined(_WIN32) || defined(_WIN64)
  static std::string GB2312ToUTF_8(const char *gb2312);
  static std::string UTF_8ToGB2312(const char *text);
#endif
};
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CORE_UTILS_FILE_UTILS_H_
