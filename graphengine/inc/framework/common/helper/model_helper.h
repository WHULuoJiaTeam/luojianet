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

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_

#include <memory>
#include <string>

#include "framework/common/fmk_types.h"
#include "framework/common/helper/om_file_helper.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "framework/common/types.h"
#include "graph/model.h"

namespace ge {
class GE_FUNC_VISIBILITY ModelHelper {
 public:
  ModelHelper() = default;
  ~ModelHelper();

  Status SaveToOmModel(const GeModelPtr &ge_model, const SaveParam &save_param, const std::string &output_file,
                       ge::ModelBufferData &model);
  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const SaveParam &save_param,
                           const std::string &output_file, ModelBufferData &model, const bool is_unknown_shape);
  Status SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file);
  Status LoadModel(const ge::ModelData &model_data);
  Status LoadRootModel(const ge::ModelData &model_data);
  static void SetModelToGeModel(GeModelPtr &ge_model, Model &model);

  GeModelPtr GetGeModel();
  GeRootModelPtr GetGeRootModel();
  void SetSaveMode(const bool val) {
    is_offline_ = val;
  }

  bool GetModelType() const {
    return is_unknown_shape_model_;
  }

  Status GetBaseNameFromFileName(const std::string &file_name, std::string &base_name) const;
  Status GetModelNameFromMergedGraphName(const std::string &graph_name, std::string &model_name) const;

 private:
  bool is_assign_model_ = false;
  bool is_offline_ = true;
  bool is_unknown_shape_model_ = false;
  ModelFileHeader *file_header_ = nullptr;
  GeModelPtr model_;
  GeRootModelPtr root_model_;

  ModelHelper(const ModelHelper &) = default;
  ModelHelper &operator=(const ModelHelper &) = default;
  Status GenerateGeModel(OmFileLoadHelper &om_load_helper);
  Status GenerateGeRootModel(OmFileLoadHelper &om_load_helper);
  Status LoadModelData(OmFileLoadHelper &om_load_helper);
  Status LoadModelData(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, const size_t mode_index) const;
  Status LoadWeights(OmFileLoadHelper &om_load_helper);
  Status LoadWeights(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, const size_t mode_index) const;
  Status LoadTask(OmFileLoadHelper &om_load_helper);
  Status LoadTask(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, const size_t mode_index) const;
  Status LoadTBEKernelStore(OmFileLoadHelper &om_load_helper);
  Status LoadTBEKernelStore(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, const size_t mode_index) const;
  Status LoadCustAICPUKernelStore(OmFileLoadHelper &om_load_helper);
  Status LoadCustAICPUKernelStore(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model,
                                  const size_t mode_index) const;

  Status SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const ModelPartitionType type,
                            const uint8_t *const data, const size_t size, const size_t model_index) const;
  Status SaveModelDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                      Buffer &model_buffer, const size_t model_index = 0U) const;
  Status SaveSizeToModelDef(const GeModelPtr &ge_model) const;
  Status SaveModelWeights(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          const size_t model_index = 0U) const;
  Status SaveModelTbeKernel(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  Status SaveModelCustAICPU(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  Status SaveModelTaskDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          Buffer &task_buffer, const size_t model_index = 0U) const;
  Status SaveModelHeader(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                         const size_t model_num = 1U) const;
  Status SaveAllModelPartiton(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              Buffer &model_buffer, Buffer &task_buffer, const size_t model_index = 0U) const;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
