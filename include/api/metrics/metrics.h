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
#ifndef LUOJIANET_INCLUDE_API_METRICS_METRICS_H
#define LUOJIANET_INCLUDE_API_METRICS_METRICS_H
#include <vector>
#include "include/api/model.h"

namespace luojianet_ms {

class MetricsImpl;
class ModelImpl;
class MSTensor;

class Metrics {
 public:
  virtual ~Metrics() = default;
  virtual void Clear() {}
  virtual float Eval() { return 0.0; }
  virtual void Update(std::vector<MSTensor *> inputs, std::vector<MSTensor *> outputs) {}
 protected:
  friend class Model;
  friend class ModelImpl;
  MetricsImpl* metrics_impl_;
};

}  // namespace luojianet_ms
#endif  // LUOJIANET_INCLUDE_API_METRICS_METRICS_H
