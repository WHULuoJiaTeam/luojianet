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
#ifndef LUOJIANET_MS_LITE_INCLUDE_TRAIN_ACCURACY_MONITOR_H_
#define LUOJIANET_MS_LITE_INCLUDE_TRAIN_ACCURACY_MONITOR_H_
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "include/train/train_loop.h"

namespace luojianet_ms {
namespace lite {

class AccuracyMonitor : public session::TrainLoopCallBack {
 public:
  explicit AccuracyMonitor(luojianet_ms::dataset::Dataset *dataset, int check_every_n, int max_steps = -1)
      : ds_(dataset), check_every_n_(check_every_n), max_steps_(max_steps) {}
  ~AccuracyMonitor() = default;
  void Begin(const session::TrainLoopCallBackData &cb_data) override;
  int EpochEnd(const luojianet_ms::session::TrainLoopCallBackData &cb_data) override;
  const std::vector<GraphPoint> &GetAccuracyPoints() const { return accuracies_; }

 private:
  luojianet_ms::dataset::Dataset *ds_;
  std::vector<GraphPoint> accuracies_;
  int check_every_n_;
  int max_steps_;
};

}  // namespace lite
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_LITE_INCLUDE_TRAIN_ACCURACY_MONITOR_H_
