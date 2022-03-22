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

#include "framework/common/fmk_error_codes.h"

namespace domi {
StatusFactory *StatusFactory::Instance() {
  static StatusFactory instance;
  return &instance;
}
void StatusFactory::RegisterErrorNo(uint32_t err, const std::string &desc) {
  if (err_desc_.find(err) != err_desc_.end()) {
    return;
  }
  err_desc_[err] = desc;
}

std::string StatusFactory::GetErrDesc(uint32_t err) {
  auto iter_find = err_desc_.find(err);
  if (iter_find == err_desc_.end()) {
    return "";
  }
  return iter_find->second;
}
// General error code
DEF_ERRORNO(SUCCESS, "Success");
DEF_ERRORNO(FAILED, "Failed");

// Common errocode
DEF_ERRORNO(MEMALLOC_FAILED, "Failed to allocate memory!");  // 50331648
DEF_ERRORNO(PARAM_INVALID, "Parameter's invalid!");          // 50331649
DEF_ERRORNO(CCE_FAILED, "Failed to call CCE API!");          // 50331650
DEF_ERRORNO(RT_FAILED, "Failed to call runtime API!");       // 50331651
DEF_ERRORNO(INTERNAL_ERROR, "Internal errors");              // 50331652
DEF_ERRORNO(CSEC_ERROR, "Failed to call libc_sec API!");     // 50331653
DEF_ERRORNO(TEE_ERROR, "Failed to call tee API!");           // 50331653
DEF_ERRORNO(UNSUPPORTED, "Parameter's unsupported!");
DEF_ERRORNO(OUT_OF_MEMORY, "Out of memory!");

// errorcode
DEF_ERRORNO(PARSE_MODEL_FAILED, "Failed to parse the model!");
DEF_ERRORNO(PARSE_WEIGHTS_FAILED, "Failed to parse the weights!");
DEF_ERRORNO(NOT_INITIALIZED, "It hasn't been initialized!");
DEF_ERRORNO(TIMEOUT, "Running time out!");

// errorcode
DEF_ERRORNO(MODEL_NOT_READY, "The model is not ready yet!");
DEF_ERRORNO(PUSH_DATA_FAILED, "Failed to push data!");
DEF_ERRORNO(DATA_QUEUE_ISFULL, "Data queue is full!");
}  // namespace domi
