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

#ifndef LUOJIANET_MS_CORE_MINDAPI_IMPL_HELPER_H_
#define LUOJIANET_MS_CORE_MINDAPI_IMPL_HELPER_H_

#include <memory>
#include <vector>
#include <type_traits>
#include "mindapi/base/base.h"

namespace luojianet_ms::api {
template <typename T, typename U>
T &ToRef(const std::shared_ptr<U> &ptr) {
  return static_cast<T &>(*ptr);
}

template <typename T, typename U, typename = typename std::enable_if_t<std::is_base_of_v<luojianet_ms::Base, T>>,
          typename = typename std::enable_if_t<std::is_base_of_v<Base, U>>>
std::shared_ptr<T> ToImpl(const SharedPtr<U> &wrapper) {
  if (wrapper == nullptr || wrapper->impl() == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<T>(wrapper->impl());
}

template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Base, T>>>
SharedPtr<T> ToWrapper(const std::shared_ptr<luojianet_ms::Base> &impl) {
  if (impl == nullptr) {
    return nullptr;
  }
  return MakeShared<T>(impl);
}

template <typename T, typename U>
std::vector<std::shared_ptr<T>> ToImplVector(const U &wrapper_vector) {
  std::vector<std::shared_ptr<T>> impl_vector;
  impl_vector.reserve(wrapper_vector.size());
  for (auto &wrapper : wrapper_vector) {
    impl_vector.emplace_back(ToImpl<T>(wrapper));
  }
  return impl_vector;
}

template <typename T, typename U>
std::vector<SharedPtr<T>> ToWrapperVector(const U &impl_vector) {
  std::vector<SharedPtr<T>> wrapper_vector;
  wrapper_vector.reserve(impl_vector.size());
  for (auto &impl : impl_vector) {
    wrapper_vector.emplace_back(ToWrapper<T>(impl));
  }
  return wrapper_vector;
}

#define MIND_API_BASE_IMPL(current_class, impl_class, base_class)                                 \
  current_class::current_class(const std::shared_ptr<luojianet_ms::Base> &impl) : base_class(impl) { \
    if (!impl_->isa<impl_class>()) {                                                              \
      MS_LOG(EXCEPTION) << "Wrong impl " << impl_->type_name() << " for " << #current_class;      \
    }                                                                                             \
  }                                                                                               \
  uint32_t current_class::ClassId() { return impl_class::kTypeId; }
}  // namespace luojianet_ms::api
#endif  // LUOJIANET_MS_CORE_MINDAPI_IMPL_HELPER_H_
