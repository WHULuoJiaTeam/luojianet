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

#ifndef LUOJIANET_MS_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
#define LUOJIANET_MS_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "utils/hash_map.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/strategy.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"

namespace luojianet_ms {
namespace parallel {
using StrategyMap = luojianet_ms::HashMap<std::string, StrategyPtr>;
using TensorLayoutPtr = std::shared_ptr<TensorLayout>;
using TensorInfoMap = luojianet_ms::HashMap<std::string, TensorLayoutPtr>;
using ParameterMap = std::vector<std::pair<std::string, ParameterPtr>>;
using ManualShapeMap = luojianet_ms::HashMap<std::string, std::vector<std::pair<int64_t, int64_t>>>;
using GroupInfoMap = std::vector<std::pair<std::string, std::vector<uint32_t>>>;
class StrategyCheckpoint {
 public:
  StrategyCheckpoint() {
    current_stage_ = 0;
    load_file_ = "";
    load_checkpoint_on_ = false;
    save_file_ = "";
    save_checkpoint_on_ = false;
    group_info_save_file_ = "";
    group_info_save_on_ = false;
  }
  ~StrategyCheckpoint() = default;

  Status Load(StrategyMap *strategy_map);
  Status LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map);
  Status Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map, ManualShapeMap *manual_shape_map);
  Status SaveGroupInfo(const GroupInfoMap &group_info_map, const RankList &restore_rank_list);
  bool group_info_save_on() const { return group_info_save_on_; }

  static StrategyCheckpoint &GetInstance();
  bool LoadCheckPointOn() const { return load_checkpoint_on_; }
  bool SaveCheckPointOn() const { return save_checkpoint_on_; }

 private:
  std::string load_file_;
  std::string save_file_;
  bool load_checkpoint_on_;
  bool save_checkpoint_on_;
  bool CheckPointExit(const std::string path) const;
  bool CheckPath(const std::string path) const;
  int64_t current_stage_;
  std::string group_info_save_file_;
  bool group_info_save_on_;
};
}  // namespace parallel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_FRONTEND_PARALLEL_STRATEGY_CHEKCPOINT_PARALLEL_STRATEGY_CHECKPOINT_H_
