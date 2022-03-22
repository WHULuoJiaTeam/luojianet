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

#ifndef HF25A2E8D_A775_44BF_88D1_166DFF56186A
#define HF25A2E8D_A775_44BF_88D1_166DFF56186A

#include <type_traits>
#include "easy_graph/eg.h"

EG_NS_BEGIN

template<typename T, typename... TS>
using all_same_traits = typename std::enable_if<std::conjunction<std::is_same<T, TS>...>::value>::type;

template<typename T, typename... TS>
using all_same_but_none_traits = typename std::enable_if<
    std::disjunction<std::bool_constant<not(sizeof...(TS))>, std::conjunction<std::is_same<T, TS>...>>::value>::type;

#define ALL_SAME_CONCEPT(TS, T) all_same_traits<T, TS...> * = nullptr
#define ALL_SAME_BUT_NONE_CONCEPT(TS, T) ::EG_NS::all_same_but_none_traits<T, TS...> * = nullptr

#define SUBGRAPH_CONCEPT(GS, G) ALL_SAME_BUT_NONE_CONCEPT(GS, G)

EG_NS_END

#endif
