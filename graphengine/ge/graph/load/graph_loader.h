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

#ifndef GE_GRAPH_LOAD_GRAPH_LOADER_H_
#define GE_GRAPH_LOAD_GRAPH_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/log.h"
#include "framework/common/fmk_types.h"
#include "framework/common/ge_types.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/model.h"
#include "runtime/mem.h"

namespace ge {
class GraphLoader {
 public:
  GraphLoader() = default;

  virtual ~GraphLoader() = default;

  GraphLoader(const GraphLoader &in) = delete;

  GraphLoader &operator=(const GraphLoader &in) = delete;

  static Status UnloadModel(uint32_t model_id);

  static Status GetMaxUsedMemory(uint32_t model_id, uint64_t &max_size);

  static Status CommandHandle(const Command &command);

  static Status GetMemoryInfo(int64_t &free);

  static Status LoadDataFromFile(const std::string &path, int32_t priority, ModelData &model_data);

  static Status LoadModelFromData(uint32_t &model_id, const ModelData &model_data, void *dev_ptr, size_t mem_size,
                                  void *weight_ptr, size_t weight_size);

  static Status LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                               const std::vector<uint32_t> &input_queue_ids,
                               const std::vector<uint32_t> &output_queue_ids);

  static Status ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                             const std::vector<GeTensorDesc> &input_desc, OutputData &output_data,
                             std::vector<GeTensorDesc> &output_desc);

  static Status DestroyAicpuKernel(uint64_t session_id, uint32_t model_id, uint32_t sub_model_id);

  static Status DestroyAicpuSessionForInfer(uint32_t model_id);

  static Status LoadModelOnline(uint32_t &model_id, const std::shared_ptr<ge::GeRootModel> &ge_root_model,
                                const std::shared_ptr<ModelListener> &listener);
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_GRAPH_LOADER_H_
