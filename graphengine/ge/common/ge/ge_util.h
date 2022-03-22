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

#ifndef GE_COMMON_GE_GE_UTIL_H_
#define GE_COMMON_GE_GE_UTIL_H_

#include <iostream>
#include <memory>
#include <utility>

namespace ge {
#define GE_DELETE_ASSIGN_AND_COPY(Classname)        \
  Classname &operator=(const Classname &) = delete; \
  Classname(const Classname &) = delete;

template <typename T, typename... Args>
static inline std::shared_ptr<T> MakeShared(Args &&... args) {
  typedef typename std::remove_const<T>::type T_nc;
  std::shared_ptr<T> ret(new (std::nothrow) T_nc(std::forward<Args>(args)...));
  return ret;
}
}  // namespace ge
#endif  // GE_COMMON_GE_GE_UTIL_H_
