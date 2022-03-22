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

#ifndef COMMON_GRAPH_UTILS_MEM_UTILS_H_
#define COMMON_GRAPH_UTILS_MEM_UTILS_H_

#include <memory>
#include <utility>

namespace ge {
template <typename _Tp, typename... _Args>
static inline std::shared_ptr<_Tp> MakeShared(_Args &&... __args) {
  typedef typename std::remove_const<_Tp>::type _Tp_nc;
  const std::shared_ptr<_Tp> ret(new (std::nothrow) _Tp_nc(std::forward<_Args>(__args)...));
  return ret;
}
}

#endif  // COMMON_GRAPH_UTILS_MEM_UTILS_H_
