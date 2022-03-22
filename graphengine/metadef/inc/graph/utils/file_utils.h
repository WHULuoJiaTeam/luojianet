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


#ifndef COMMON_GRAPH_UTILS_FILE_UTILS_H_
#define COMMON_GRAPH_UTILS_FILE_UTILS_H_

#include <cstdint>
#include <string>
#include "external/graph/types.h"

namespace ge {
///
/// @ingroup domi_common
/// @brief Absolute path for obtaining files.
/// @param [in] path of input file
/// @param [out] Absolute path of a file. If the absolute path cannot be obtained, an empty string is returned
///
std::string RealPath(const char_t *path);

///
/// @ingroup domi_common
/// @brief Recursively Creating a Directory
/// @param [in] directory_path  Path, which can be a multi-level directory.
/// @return 0 success
/// @return -1 fail
///
int32_t CreateDirectory(const std::string &directory_path);

}

#endif // end COMMON_GRAPH_UTILS_FILE_UTILS_H_
