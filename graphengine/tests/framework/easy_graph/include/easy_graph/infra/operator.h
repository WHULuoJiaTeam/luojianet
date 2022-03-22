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

#ifndef H05B2224D_B927_4FC0_A936_97B52B8A99DB
#define H05B2224D_B927_4FC0_A936_97B52B8A99DB

//////////////////////////////////////////////////////////////
#define __DECL_EQUALS(cls)                                                                                             \
  bool operator!=(const cls &rhs) const;                                                                               \
  bool operator==(const cls &rhs) const

//////////////////////////////////////////////////////////////
#define __FIELD_EQ(name) this->name == rhs.name
#define __FIELD_LT(name) this->name < rhs.name

//////////////////////////////////////////////////////////////
#define __SUPER_EQ(super) static_cast<const super &>(*this) == rhs
#define __SUPER_LT(super) static_cast<const super &>(*this) < rhs

//////////////////////////////////////////////////////////////
#define __DEF_EQUALS(cls)                                                                                              \
  bool cls::operator!=(const cls &rhs) const {                                                                         \
    return !(*this == rhs);                                                                                            \
  }                                                                                                                    \
  bool cls::operator==(const cls &rhs) const

/////////////////////////////////////////////////////////////
#define __INLINE_EQUALS(cls)                                                                                           \
  bool operator!=(const cls &rhs) const {                                                                              \
    return !(*this == rhs);                                                                                            \
  }                                                                                                                    \
  bool operator==(const cls &rhs) const

/////////////////////////////////////////////////////////////
#define __DECL_COMP(cls)                                                                                               \
  __DECL_EQUALS(cls);                                                                                                  \
  bool operator<(const cls &) const;                                                                                   \
  bool operator>(const cls &) const;                                                                                   \
  bool operator<=(const cls &) const;                                                                                  \
  bool operator>=(const cls &) const

/////////////////////////////////////////////////////////////
#define __DEF_COMP(cls)                                                                                                \
  bool cls::operator>(const cls &rhs) const {                                                                          \
    return !(*this <= rhs);                                                                                            \
  }                                                                                                                    \
  bool cls::operator>=(const cls &rhs) const {                                                                         \
    return !(*this < rhs);                                                                                             \
  }                                                                                                                    \
  bool cls::operator<=(const cls &rhs) const {                                                                         \
    return (*this < rhs) || (*this == rhs);                                                                            \
  }                                                                                                                    \
  bool cls::operator<(const cls &rhs) const

#endif
