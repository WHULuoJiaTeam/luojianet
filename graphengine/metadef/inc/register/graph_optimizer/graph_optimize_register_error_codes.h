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

#ifndef INC_REGISTER_GRAPH_OPTIMIZE_REGISTER_ERROR_CODES_H_
#define INC_REGISTER_GRAPH_OPTIMIZE_REGISTER_ERROR_CODES_H_

#include <map>
#include <string>

/** Assigned SYS ID */
const uint8_t SYSID_FE = 3;

/** Common module ID */
const uint8_t FE_MODID_COMMON = 50;

namespace fe {

/**  FE error code definiton Macro
*  Build error code
*/
#define FE_DEF_ERRORNO(sysid, modid, name, value, desc)                            \
  static constexpr fe::Status name =                                               \
      ((((static_cast<uint32_t>((0xFF) & (static_cast<uint8_t>(sysid)))) << 24) |  \
       ((static_cast<uint32_t>((0xFF) & (static_cast<uint8_t>(modid)))) << 16)) |  \
       ((0xFFFF) & (static_cast<uint16_t>(value))));

using Status = uint32_t;

#define FE_DEF_ERRORNO_COMMON(name, value, desc)                  \
  FE_DEF_ERRORNO(SYSID_FE, FE_MODID_COMMON, name, value, desc)

using Status = uint32_t;

FE_DEF_ERRORNO(0, 0, SUCCESS, 0, "success");
FE_DEF_ERRORNO(0xFF, 0xFF, FAILED, 0xFFFF, "failed");
FE_DEF_ERRORNO_COMMON(NOT_CHANGED, 201, "The nodes of the graph not changed.");
FE_DEF_ERRORNO_COMMON(PARAM_INVALID, 1, "Parameter's invalid!");
FE_DEF_ERRORNO_COMMON(GRAPH_FUSION_CYCLE, 301, "Graph is cycle after fusion!");

}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZE_REGISTER_ERROR_CODES_H_
