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

#ifndef INC_FRAMEWORK_COMMON_FMK_ERROR_CODES_H_
#define INC_FRAMEWORK_COMMON_FMK_ERROR_CODES_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#define GE_OBJECT_VISIBILITY
#else
#define GE_FUNC_VISIBILITY
#define GE_OBJECT_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#define GE_OBJECT_VISIBILITY
#else
#define GE_FUNC_VISIBILITY
#define GE_OBJECT_VISIBILITY __attribute__((visibility("hidden")))
#endif
#endif

#include <map>
#include <string>

#include "framework/common/fmk_types.h"
#include "register/register_error_codes.h"

// Each module uses the following four macros to define error codes:
#define DECLARE_ERRORNO_OMG(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_OMG, name, value)
#define DECLARE_ERRORNO_OME(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_OME, name, value)
#define DECLARE_ERRORNO_CALIBRATION(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_CALIBRATION, name, value)

#define DEF_ERRORNO(name, desc) const ErrorNoRegisterar g_##name##_errorno(name, desc);

// Interface for Obtaining Error Code Description
#define GET_ERRORNO_STR(value) domi::StatusFactory::Instance()->GetErrDesc(value)

const int MODID_OMG = 1;          // OMG module ID
const int MODID_OME = 2;          // OME module ID
const int MODID_CALIBRATION = 3;  // Calibration module ID

namespace domi {
class GE_FUNC_VISIBILITY StatusFactory {
 public:
  static StatusFactory *Instance();

  void RegisterErrorNo(uint32_t err, const std::string &desc);

  std::string GetErrDesc(uint32_t err);

 protected:
  StatusFactory() {}
  ~StatusFactory() {}

 private:
  std::map<uint32_t, std::string> err_desc_;
};

class GE_FUNC_VISIBILITY ErrorNoRegisterar {
 public:
  ErrorNoRegisterar(uint32_t err, const std::string &desc) { StatusFactory::Instance()->RegisterErrorNo(err, desc); }
  ~ErrorNoRegisterar() {}
};

// Common errocode
DECLARE_ERRORNO_COMMON(MEMALLOC_FAILED, 0);  // 50331648
DECLARE_ERRORNO_COMMON(CCE_FAILED, 2);       // 50331650
DECLARE_ERRORNO_COMMON(RT_FAILED, 3);        // 50331651
DECLARE_ERRORNO_COMMON(INTERNAL_ERROR, 4);   // 50331652
DECLARE_ERRORNO_COMMON(CSEC_ERROR, 5);       // 50331653
DECLARE_ERRORNO_COMMON(TEE_ERROR, 6);        // 50331653
DECLARE_ERRORNO_COMMON(UNSUPPORTED, 100);
DECLARE_ERRORNO_COMMON(OUT_OF_MEMORY, 101);

// Omg errorcode
DECLARE_ERRORNO_OMG(PARSE_MODEL_FAILED, 0);
DECLARE_ERRORNO_OMG(PARSE_WEIGHTS_FAILED, 1);
DECLARE_ERRORNO_OMG(NOT_INITIALIZED, 2);
DECLARE_ERRORNO_OMG(TIMEOUT, 3);

// Ome errorcode
DECLARE_ERRORNO_OME(MODEL_NOT_READY, 0);
DECLARE_ERRORNO_OME(PUSH_DATA_FAILED, 1);
DECLARE_ERRORNO_OME(DATA_QUEUE_ISFULL, 2);
}  // namespace domi

#endif  // INC_FRAMEWORK_COMMON_FMK_ERROR_CODES_H_
