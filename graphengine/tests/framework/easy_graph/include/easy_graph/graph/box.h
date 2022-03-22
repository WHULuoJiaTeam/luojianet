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

#ifndef H4AA49861_3311_4114_8687_1C7D04FA43B9
#define H4AA49861_3311_4114_8687_1C7D04FA43B9

#include <memory>
#include "easy_graph/infra/keywords.h"

EG_NS_BEGIN

INTERFACE(Box){};

using BoxPtr = std::shared_ptr<Box>;

template<typename Anything, typename... Args>
BoxPtr BoxPacking(Args &&... args) {
  return std::make_shared<Anything>(std::forward<Args>(args)...);
}

template<typename Anything>
Anything *BoxUnpacking(const BoxPtr &box) {
  return dynamic_cast<Anything *>(box.get());
}

EG_NS_END

#endif
