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

#ifndef H05B2224D_B926_4FC0_A936_77B52B8A98DB
#define H05B2224D_B926_4FC0_A936_77B52B8A98DB

#include "easy_graph/eg.h"

EG_NS_BEGIN

namespace details {
template<typename T>
struct DefaultValue {
  static T value() {
    return T();
  }
};

template<typename T>
struct DefaultValue<T *> {
  static T *value() {
    return 0;
  }
};

template<typename T>
struct DefaultValue<const T *> {
  static T *value() {
    return 0;
  }
};

template<>
struct DefaultValue<void> {
  static void value() {}
};
}  // namespace details

#define DEFAULT(type, method)                                                                                          \
  virtual type method {                                                                                                \
    return ::EG_NS::details::DefaultValue<type>::value();                                                              \
  }

EG_NS_END

#endif
