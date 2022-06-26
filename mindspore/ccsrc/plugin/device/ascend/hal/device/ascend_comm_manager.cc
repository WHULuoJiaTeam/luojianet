/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/common/utils/comm_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "hccl/hcom.h"
#include "plugin/device/ascend/hal/device/distribute/ascend_collective.h"
#include "utils/ms_context.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace {
#define HCCL_RUN_CHECK(op_name, group, op)                      \
  do {                                                          \
    auto hccl_result = static_cast<int64_t>(op);                \
    if (hccl_result != 0) {                                     \
      MS_LOG(ERROR) << op_name << " failed: #" << group << "#"; \
      return false;                                             \
    }                                                           \
  } while (0)

#define HCCL_GROUP_CHECK_EMPTY(group)                              \
  do {                                                             \
    if (group.length() == 0) {                                     \
      MS_LOG(ERROR) << "The length of group name should not be 0"; \
      return false;                                                \
    }                                                              \
  } while (0)

#define HCCL_GROUP_CHECK_IS_WORLD(group)                                   \
  do {                                                                     \
    if (group == kHcclWorldGroup) {                                        \
      MS_LOG(ERROR) << "The group name should not be " << kHcclWorldGroup; \
      return false;                                                        \
    }                                                                      \
  } while (0)

class AscendCommManager : public CommManager {
 public:
  AscendCommManager() : CommManager("hccl") {}
  ~AscendCommManager() override = default;

  bool CreateGroupSync(const string &group, const std::vector<unsigned int> &rank_id_list) const override {
    auto rank_size = rank_id_list.size();
    HCCL_GROUP_CHECK_EMPTY(group);
    HCCL_GROUP_CHECK_IS_WORLD(group);
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    bool is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
    auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
    if (!is_task_sink && mode == kGraphMode) {
      HcclCollectiveGroup::instance().CreateCommGroup(group, rank_id_list);
    } else {
      HCCL_RUN_CHECK(string("create communicate group"), group,
                     hccl::HcclAdapter::GetInstance().HcclCreateGroup(group, UlongToUint(rank_size),
                                                                      std::vector<unsigned int>(rank_id_list).data()));
    }
    return true;
  }

  bool GetRankID(const string &group, unsigned int *rank_id) const override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
      HCCL_GROUP_CHECK_EMPTY(group);
      if (!context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
        *rank_id = static_cast<unsigned int>(HcclCollectiveGroup::instance().GetRankId(group));
      } else {
        HCCL_RUN_CHECK(string("get rank_id"), group, hccl::HcclAdapter::GetInstance().HcclGetRankId(group, rank_id));
      }
    } else {
      HCCL_RUN_CHECK(string("get rank_id"), group, hccl::HcclAdapter::GetInstance().HcclGetRankId(rank_id));
    }
    return true;
  }

  bool GetRankSize(const string &group, unsigned int *rank_size) const override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
      HCCL_GROUP_CHECK_EMPTY(group);
      if (!context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
        *rank_size = static_cast<unsigned int>(HcclCollectiveGroup::instance().GetRankSize(group));
      } else {
        HCCL_RUN_CHECK(string("get rank size"), group,
                       hccl::HcclAdapter::GetInstance().HcclGetRankSize(group, rank_size));
      }
    } else {
      HCCL_RUN_CHECK(string("get rank size"), group, hccl::HcclAdapter::GetInstance().HcclGetRankSize(rank_size));
    }
    return true;
  }

  bool DestroyGroup(const string &group) const override {
    HCCL_GROUP_CHECK_EMPTY(group);
    HCCL_GROUP_CHECK_IS_WORLD(group);
    HCCL_RUN_CHECK(string("destroy communicate group"), group,
                   hccl::HcclAdapter::GetInstance().HcclDestroyGroup(group));
    return true;
  }

  uint32_t GetRank() override {
    uint32_t rank_id = 0;
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    if (parallel_context->parallel_mode() != parallel::kStandalone) {
      // Check HCCL inited.
      if (!hccl::HcclAdapter::GetInstance().Inited()) {
        MS_LOG(DEBUG) << "HCCL not inited, return rank_id = 0";
        return rank_id;
      }
      if (!GetRankID(kHcclWorldGroup, &rank_id)) {
        MS_LOG(EXCEPTION) << "Get rank id failed.";
      }
    }
    return rank_id;
  }
};
COMM_MANAGER_REG(kAscendDevice, std::make_shared<AscendCommManager>());
}  // namespace
}  // namespace mindspore
