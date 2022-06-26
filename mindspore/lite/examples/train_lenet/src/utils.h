/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_EXAMPLES_TRAIN_LENET_SRC_UTILS_H_
#define MINDSPORE_LITE_EXAMPLES_TRAIN_LENET_SRC_UTILS_H_

// DEBUG should be defined because the source code use assert to test exception
// but the Code Analysis Tool not allow us to use assert directly
#define DEBUG TRUE

#ifdef DEBUG
#include <cassert>
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif

#endif  // MINDSPORE_LITE_EXAMPLES_TRAIN_LENET_SRC_UTILS_H_
