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

#ifndef GE_SINGLE_OP_SINGLE_OP_MANAGER_H_
#define GE_SINGLE_OP_SINGLE_OP_MANAGER_H_

#include <mutex>
#include <unordered_map>
#include <string>
#include "common/ge/op_tiling_manager.h"
#include "single_op/single_op_model.h"
#include "single_op/stream_resource.h"

namespace ge {
class SingleOpManager {
 public:
  ~SingleOpManager();

  static SingleOpManager &GetInstance() {
    static SingleOpManager instance;
    return instance;
  }

  Status GetOpFromModel(const std::string &model_name,
                        const ge::ModelData &model_data,
                        void *stream,
                        SingleOp **single_op,
                        const uint64_t model_id);

  Status GetDynamicOpFromModel(const std::string &model_name,
                               const ge::ModelData &model_data,
                               void *stream,
                               DynamicSingleOp **dynamic_single_op,
                               const uint64_t model_id);

  StreamResource *GetResource(uintptr_t resource_id, rtStream_t stream);

  Status ReleaseResource(void *stream);

  void RegisterTilingFunc();

 private:
  static Status GetResourceId(rtStream_t stream, uintptr_t &resource_id);

  StreamResource *TryGetResource(uintptr_t resource_id);

  std::mutex mutex_;
  bool tiling_func_registered_ = false;
  std::unordered_map<uintptr_t, StreamResource *> stream_resources_;
  OpTilingManager op_tiling_manager_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_SINGLE_OP_MANAGER_H_
