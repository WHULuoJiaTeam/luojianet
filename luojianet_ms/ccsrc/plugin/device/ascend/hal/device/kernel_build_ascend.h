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

#ifndef LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_BUILD_ASCEND_H_
#define LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_BUILD_ASCEND_H_

#include <vector>
#include <map>
#include "backend/common/session/kernel_graph.h"

namespace luojianet_ms {
namespace device {
namespace ascend {
using CommOpInputInfo = std::map<AnfNodePtr, std::vector<size_t>>;
using CleanOpsMap = std::map<CNodePtr, std::vector<CNodePtr>>;

/**
 * @brief kernel build for ascend.
 */
bool KernelBuild(const std::vector<CNodePtr> &kernels);

/**
 * @brief insert atomic
 */
void InsertAtomicCleanOps(const KernelGraphPtr &kernel_graph);

/**
 *  @brief insert atomic for mind rt
 * */
void InsertAtomicCleanOps(const std::vector<CNodePtr> &nodes, CleanOpsMap *maps);
}  // namespace ascend
}  // namespace device
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_BUILD_ASCEND_H_
