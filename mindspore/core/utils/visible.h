/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_VISIBLE_H_
#define MINDSPORE_CORE_UTILS_VISIBLE_H_

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef BUILDING_DLL
#define MS_CORE_API __declspec(dllexport)
#define MS_EXPORT __declspec(dllexport)
#define GVAR_DEF(type, name, value) MS_CORE_API inline const type name = value;
#else
#define MS_CORE_API __declspec(dllimport)
#define MS_EXPORT __declspec(dllimport)
#define GVAR_DEF(type, name, value) MS_CORE_API extern const type name;
#endif
#define MS_LOCAL
#else
#define MS_CORE_API __attribute__((visibility("default")))
#define MS_EXPORT __attribute__((visibility("default")))
#define MS_LOCAL __attribute__((visibility("hidden")))
#define GVAR_DEF(type, name, value) MS_CORE_API inline const type name = value;
#endif
#endif  // MINDSPORE_CORE_UTILS_VISIBLE_H_
