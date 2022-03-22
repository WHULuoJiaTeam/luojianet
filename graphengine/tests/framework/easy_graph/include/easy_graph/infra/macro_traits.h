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

#ifndef H7DA4A075_246A_4DD6_B1BB_ECA3806C9483
#define H7DA4A075_246A_4DD6_B1BB_ECA3806C9483

#include "easy_graph/eg.h"

EG_NS_BEGIN

////////////////////////////////////////////////////////////////////////

#define VA_ARGS_NUM(...)                                                                                               \
  VA_ARGS_NUM_PRIVATE(0, ##__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,    \
                      45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22,  \
                      21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define VA_ARGS_NUM_PRIVATE(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19,  \
                            _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37,  \
                            _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55,  \
                            _56, _57, _58, _59, _60, _61, _62, _63, _64, N, ...)                                       \
  N

////////////////////////////////////////////////////////////////////////

#define __MACRO_CONCAT(x, y) x##y
#define MACRO_CONCAT(x, y) __MACRO_CONCAT(x, y)

#define __MACRO_SECOND(FIRST, SECOND, ...) SECOND
#define MACRO_SECOND(...) __MACRO_SECOND(__VA_ARGS__)

#define MACRO_VERIFY_FIRST(...) MACRO_SECOND(__VA_ARGS__, 1)

#define MACRO_BOOL_0 MACRO_DUMMY, 0
#define MACRO_BOOL(N) MACRO_VERIFY_FIRST(__MACRO_CONCAT(MACRO_BOOL_, N))

#define MACRO_CONDITION_0(TRUE_BRANCH, FALSE_BRANCH) FALSE_BRANCH
#define MACRO_CONDITION_1(TRUE_BRANCH, FALSE_BRANCH) TRUE_BRANCH
#define MACRO_CONDITION(N) MACRO_CONCAT(MACRO_CONDITION_, MACRO_BOOL(N))

#define NOT_EMPTY_SELECT(...) MACRO_CONDITION(VA_ARGS_NUM(__VA_ARGS__))

EG_NS_END

#endif
