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

#ifndef HE7D53295_59F0_46B0_A881_D6A33B1F9C14
#define HE7D53295_59F0_46B0_A881_D6A33B1F9C14

#include <type_traits>
#include "easy_graph/graph/box.h"

EG_NS_BEGIN

namespace detail {
template<typename Anything>
struct BoxWrapper : Anything, Box {
  using Anything::Anything;
};

template<typename Anything>
using BoxedAnything = std::conditional_t<std::is_base_of_v<Box, Anything>, Anything, BoxWrapper<Anything>>;
}  // namespace detail

#define BOX_WRAPPER(Anything) ::EG_NS::detail::BoxedAnything<Anything>
#define BOX_OF(Anything, ...) ::EG_NS::BoxPacking<Anything>(__VA_ARGS__)

EG_NS_END

#endif
