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

#ifndef INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTIL_H
#define INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTIL_H

#include <map>
#include <string>
#include <vector>
#include "ge/ge_api_error_codes.h"
#include "nlohmann/json.hpp"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/ge_tensor.h"

namespace ge {
extern const int64_t kBlockSize;
extern const std::string kBinFileValues;
extern const std::string kBinIdValue;
extern const std::string kBinFilePathValue;

struct FileConstantInfo {
  std::string value_bin_file_id;
  std::string value_bin_file_path;
};

struct OptionInfo {
  std::vector<FileConstantInfo> info;
};

void from_json(const nlohmann::json &j, FileConstantInfo &info);

void from_json(const nlohmann::json &j, OptionInfo &option_info);

Status GetFilePathFromOption(std::map<std::string, std::string> &file_id_and_path_map);

Status CopyOneWeightFromFile(const void *curr_dev_ptr, const std::string &value, const size_t file_constant_size,
                             size_t &left_size);

Status GetFilePath(const OpDescPtr &op_desc, const std::map<std::string, std::string> &file_id_and_path_map,
                   std::string &file_path);

Status GetFileConstantElementTotalSize(const GeShape &shape, const DataType data_type, int64_t &mem_size,
                                       const Format format = FORMAT_ND);
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTIL_H
