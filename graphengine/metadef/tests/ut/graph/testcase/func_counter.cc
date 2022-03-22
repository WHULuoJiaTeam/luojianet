/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "func_counter.h"
namespace ge {
size_t FuncCounter::construct_times = 0;
size_t FuncCounter::copy_construct_times = 0;
size_t FuncCounter::move_construct_times = 0;
size_t FuncCounter::copy_assign_times = 0;
size_t FuncCounter::move_assign_times = 0;
size_t FuncCounter::destruct_times = 0;
}  // namespace ge