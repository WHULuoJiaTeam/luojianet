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

#ifndef LUOJIANET_MS_CORE_MINDRT_INCLUDE_ASYNC_FAILURE_H
#define LUOJIANET_MS_CORE_MINDRT_INCLUDE_ASYNC_FAILURE_H

#include "async/status.h"

namespace luojianet_ms {

class Failure : public Status {
 public:
  Failure() : Status(Status::KOK), errorCode(Status::KOK) {}

  Failure(int32_t code) : Status(code), errorCode(code) {}

  ~Failure() {}

  const int32_t GetErrorCode() const { return errorCode; }

 private:
  Status::Code errorCode;
};

}  // namespace luojianet_ms

#endif /* __FAILURE_HPP__ */
