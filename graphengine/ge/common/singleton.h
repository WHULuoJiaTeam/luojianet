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
#ifndef GE_COMMON_SINGLETON_H_
#define GE_COMMON_SINGLETON_H_

#include <memory>

#define DECLARE_SINGLETON_CLASS(T) friend class Singleton<T>

namespace ge {
static std::mutex single_mutex_;
// Single thread version single instance template
template <typename T>
class Singleton {
 public:
  Singleton(Singleton const &) = delete;
  Singleton &operator=(Singleton const &) = delete;

  template <typename... _Args>
  static T *Instance(_Args... args) {
    std::lock_guard<std::mutex> lock(single_mutex_);
    if (instance_ == nullptr) {
      // std::nothrow, Nullptr returned when memory request failed
      instance_.reset(new (std::nothrow) T(args...));
    }
    return instance_.get();
  }

  static void Destroy(void) { instance_.reset(); }

  Singleton() = default;
  virtual ~Singleton() = default;

 private:
  static std::unique_ptr<T> instance_;
};

template <typename T>
std::unique_ptr<T> Singleton<T>::instance_;
}  // namespace ge
#endif  // GE_COMMON_SINGLETON_H_
