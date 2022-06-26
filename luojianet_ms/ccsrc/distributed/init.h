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

#ifndef LUOJIANET_MS_CCSRC_DISTRIBUTED_INIT_H_
#define LUOJIANET_MS_CCSRC_DISTRIBUTED_INIT_H_

#include <vector>
#include <string>
#include "distributed/collective/collective_manager.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "distributed/cluster/cluster_context.h"
#else
#include "distributed/cluster/dummy_cluster_context.h"
#endif
#include "include/backend/visible.h"

namespace luojianet_ms {
namespace distributed {
// The static methods of LuoJiaNET distributed execution. They can be exported by Pybind.

// Initialize and finalize distributed execution.
BACKEND_EXPORT bool Initialize();
bool Finalize();

// Initialize and finalize the cluster based on LuoJiaNET communication framework.
bool InitializeCluster();
bool FinalizeCluster();

// Initialize and finalize collective communication for distributed execution.
bool InitializeCollective();
bool FinalizeCollective();
}  // namespace distributed
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_DISTRIBUTED_INIT_H_
