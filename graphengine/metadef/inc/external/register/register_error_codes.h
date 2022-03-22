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

#ifndef INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_
#define INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_

#define SYSID_FWK 3U     // Subsystem ID
#define MODID_COMMON 0U  // Common module ID

#define DECLARE_ERRORNO(sysid, modid, name, value)                               \
  constexpr domi::Status (name) =                                                \
      ((static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(sysid)))) << 24U) | \
      ((static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(modid)))) << 16U) | \
      (static_cast<uint32_t>((0xFFFFU & (static_cast<uint32_t>(value)))));

#define DECLARE_ERRORNO_COMMON(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_COMMON, (name), (value))

namespace domi {
using Status = uint32_t;

// General error code
DECLARE_ERRORNO(0U, 0U, SUCCESS, 0U);
DECLARE_ERRORNO(0xFFU, 0xFFU, FAILED, 0xFFFFFFFFU);
DECLARE_ERRORNO_COMMON(PARAM_INVALID, 1U);  // 50331649
DECLARE_ERRORNO(SYSID_FWK, 1U, SCOPE_NOT_CHANGED, 201U);
}  // namespace domi

#endif  // INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_
