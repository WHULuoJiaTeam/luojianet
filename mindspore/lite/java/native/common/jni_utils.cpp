/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "common/jni_utils.h"
#include <cstring>
#include <cstdlib>
#include <climits>
#include <memory>
#include "common/ms_log.h"

std::string RealPath(const char *path) {
  if (path == nullptr) {
    MS_LOGE("path is nullptr");
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    MS_LOGE("path is too long");
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOGE("new resolved_path failed");
    return "";
  }
#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), path, 1024);
#else
  char *real_path = realpath(path, resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOGE("file path is not valid : %s", path);
    return "";
  }
  std::string res = resolved_path.get();
  return res;
}
