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

#include "runtime/collective/communication_group.h"

namespace mindspore {
namespace device {
CommunicationGroup::CommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                       uint32_t global_rank)
    : initialized_(false),
      global_rank_(global_rank),
      size_(group_ranks.size()),
      name_(name),
      group_ranks_(group_ranks) {
  uint32_t group_rank = 0;
  // The input group_ranks contains the global ranks of the processes in this group.
  (void)std::for_each(group_ranks.begin(), group_ranks.end(), [&](const uint32_t &global_rank) {
    global_to_group_ranks_[global_rank] = group_rank;
    group_to_global_ranks_[group_rank] = global_rank;
    group_rank++;
  });
}

uint32_t CommunicationGroup::GetGroupRank(uint32_t global_rank) {
  CHECK_RET((global_to_group_ranks_.count(global_rank) != 0), true,
            "Group " + name_ + " doesn't contain the global rank " + std::to_string(global_rank));
  return global_to_group_ranks_[global_rank];
}

uint32_t CommunicationGroup::GetGlobalRank(uint32_t group_rank) {
  CHECK_RET((group_to_global_ranks_.count(group_rank) != 0), true,
            "Group " + name_ + " doesn't contain the group rank " + std::to_string(group_rank));
  return group_to_global_ranks_[group_rank];
}

uint32_t CommunicationGroup::group_size() const { return size_; }

const std::vector<uint32_t> &CommunicationGroup::group_ranks() const { return group_ranks_; }

const std::map<uint32_t, uint32_t> &CommunicationGroup::global_to_group_ranks() const { return global_to_group_ranks_; }

const std::map<uint32_t, uint32_t> &CommunicationGroup::group_to_global_ranks() const { return group_to_global_ranks_; }
}  // namespace device
}  // namespace mindspore
