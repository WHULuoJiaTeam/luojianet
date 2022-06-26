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

#ifndef LUOJIANET_MS_CORE_MINDRT_INCLUDE_ASYNC_UUID_GENERATOR_H
#define LUOJIANET_MS_CORE_MINDRT_INCLUDE_ASYNC_UUID_GENERATOR_H

#include <string>
#include "async/uuid_base.h"

namespace luojianet_ms {
namespace uuid_generator {
struct UUID : public luojianet_ms::uuids::uuid {
 public:
  explicit UUID(const luojianet_ms::uuids::uuid &inputUUID) : luojianet_ms::uuids::uuid(inputUUID) {}
  static UUID GetRandomUUID();
  std::string ToString();
};
}  // namespace uuid_generator

namespace localid_generator {
int GenLocalActorId();

#ifdef HTTP_ENABLED
int GenHttpClientConnId();
int GenHttpServerConnId();
#endif
}  // namespace localid_generator
}  // namespace luojianet_ms
#endif
