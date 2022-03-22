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

#ifndef METADEF_CXX_FUNC_COUNTER_H
#define METADEF_CXX_FUNC_COUNTER_H
#include <iostream>
#include <vector>
namespace ge {
struct FuncCounter {
  static size_t construct_times;
  static size_t copy_construct_times;
  static size_t move_construct_times;
  static size_t copy_assign_times;
  static size_t move_assign_times;
  static size_t destruct_times;
  FuncCounter() {
    ++construct_times;
  }
  ~FuncCounter() {
    ++destruct_times;
  }
  FuncCounter(const FuncCounter &) {
    ++copy_construct_times;
  }
  FuncCounter(FuncCounter &&) noexcept {
    ++move_construct_times;
  }
  FuncCounter &operator=(const FuncCounter &) {
    ++copy_assign_times;
    return *this;
  }
  FuncCounter &operator=(FuncCounter &&) {
    ++move_assign_times;
    return *this;
  }
  static void Clear() {
    construct_times = 0;
    copy_construct_times = 0;
    copy_assign_times = 0;
    move_construct_times = 0;
    move_assign_times = 0;
    destruct_times = 0;
  }
  static std::vector<size_t> GetTimes() {
    return {construct_times,      destruct_times,    copy_construct_times,
            move_construct_times, copy_assign_times, move_assign_times};
    ;
  }
  static bool AllTimesZero() {
    if (construct_times != 0) {
      std::cout << "construct_times not 0" << std::endl;
      return false;
    }
    if (destruct_times != 0) {
      std::cout << "destruct_times not 0" << std::endl;
      return false;
    }
    if (copy_construct_times != 0) {
      std::cout << "copy_construct_times not 0" << std::endl;
      return false;
    }
    if (move_construct_times != 0) {
      std::cout << "move_construct_times not 0" << std::endl;
      return false;
    }
    if (copy_assign_times != 0) {
      std::cout << "copy_assign_times not 0" << std::endl;
      return false;
    }
    if (move_assign_times != 0) {
      std::cout << "move_assign_times not 0" << std::endl;
      return false;
    }
    return true;
  }

  static size_t GetClearConstructTimes() {
    auto tmp = construct_times;
    construct_times = 0;
    return tmp;
  }
  static size_t GetClearCopyConstructTimes() {
    auto tmp = copy_construct_times;
    copy_construct_times = 0;
    return tmp;
  }
  static size_t GetClearMoveConstructTimes() {
    auto tmp = move_construct_times;
    move_construct_times = 0;
    return tmp;
  }
  static size_t GetClearCopyAssignTimes() {
    auto tmp = copy_assign_times;
    copy_assign_times = 0;
    return tmp;
  }
  static size_t GetClearMoveAssignTimes() {
    auto tmp = move_assign_times;
    move_assign_times = 0;
    return tmp;
  }
  static size_t GetClearDestructTimes() {
    auto tmp = destruct_times;
    destruct_times = 0;
    return tmp;
  }
};

}

#endif  //METADEF_CXX_FUNC_COUNTER_H
