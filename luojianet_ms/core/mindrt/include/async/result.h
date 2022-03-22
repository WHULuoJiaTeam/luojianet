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

#ifndef LUOJIANET_MS_CORE_MINDRT_INCLUDE_ASYNC_RESULT_H
#define LUOJIANET_MS_CORE_MINDRT_INCLUDE_ASYNC_RESULT_H

#include <tuple>

#include "async/option.h"
#include "async/status.h"

namespace luojianet_ms {
template <typename... Types>
class Result {
 public:
  Result() : status(Status::KINIT) {}

  Result(Types... types, const Status &s) : tuple(Option<Types>(types)...), status(s) {}

  ~Result() {}

  template <std::size_t I>
  bool IsSome() {
    return (std::get<I>(tuple)).IsSome();
  }

  template <std::size_t I>
  bool IsNone() {
    return std::get<I>(tuple).IsNone();
  }

  bool IsOK() { return status.IsOK(); }

  bool IsError() { return status.IsError(); }

  void SetStatus(Status::Code code) { status.SetCode(code); }

  const Status &GetStatus() const { return status; }

  template <std::size_t I>
  typename std::tuple_element<I, std::tuple<Option<Types>...>>::type Get() const {
    return GetOption<I>().Get();
  }

 private:
  template <std::size_t I>
  typename std::tuple_element<I, std::tuple<Option<Types>...>>::type GetOption() const {
    return std::get<I>(tuple);
  }

 private:
  std::tuple<Option<Types>...> tuple;
  Status status;
};
}  // namespace luojianet_ms

#endif
