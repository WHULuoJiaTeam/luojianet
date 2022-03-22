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

#ifndef GE_COMMON_MODEL_SAVER_H_
#define GE_COMMON_MODEL_SAVER_H_

#include "common/auth/file_saver.h"
#include "nlohmann/json.hpp"
#include "framework/common/types.h"

/**
* Provide read and write operations for offline model files
*/
namespace ge {
using Json = nlohmann::json;

class ModelSaver : public FileSaver {
 public:
  /**
   * @ingroup domi_common
   * @brief Save JSON object to file
   * @param [in] file_path File output path
   * @param [in] model json object
   * @return Status result
   */
  static Status SaveJsonToFile(const char *file_path, const Json &model);
};
}  //  namespace ge

#endif  //  GE_COMMON_MODEL_SAVER_H_
