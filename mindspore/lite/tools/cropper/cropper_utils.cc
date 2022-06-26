/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/cropper/cropper_utils.h"
#include <fstream>
#include <iostream>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
namespace cropper {
int ValidFileSuffix(const std::string &filename, const std::string &suffix) {
  std::string suffixStr = filename.substr(filename.find_last_of('.') + 1);
  if (suffix == suffixStr) {
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "The file name suffix needs: " << suffix << ", but file is " << filename;
    return RET_ERROR;
  }
}
int ValidFile(std::ifstream &in_file, const std::string &file_path) {
  if (!in_file.good()) {
    std::cerr << "file: " << file_path << " is not exist" << std::endl;
    MS_LOG(ERROR) << "file: " << file_path << " is not exist";
    return RET_ERROR;
  }

  if (!in_file.is_open()) {
    std::cerr << "file: " << file_path << " open failed" << std::endl;
    MS_LOG(ERROR) << "file: " << file_path << " open failed";
    in_file.close();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace cropper
}  // namespace lite
}  // namespace mindspore
