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

#include "framework/common/helper/model_helper.h"

#include "common/model_parser/model_parser.h"
#include "framework/omg/model_tool.h"
#include "framework/omg/version.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "framework/omg/omg_inner_types.h"

using std::string;
using domi::ModelTaskDef;

namespace {
const int64_t kOriginalOmPartitionNum = 1;
const uint32_t kStatiOmFileModelNum = 1;
}


namespace ge {
ModelHelper::~ModelHelper() { (void)ReleaseLocalModelData(); }

Status ModelHelper::SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, ModelPartitionType type,
                                       const uint8_t *data, size_t size, size_t model_index) {
  if (size < 1 || size > UINT32_MAX) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, partition size %zu invalid", size);
    if (size > UINT32_MAX) {
      string item = "item";
      if (type == MODEL_DEF) {
        item = "model info";
      } else if (type == WEIGHTS_DATA) {
        item = "weight data";
      } else if (type == TASK_INFO) {
        item = "task info";
      } else if (type == TBE_KERNELS) {
        item = "tbe kernels";
      } else if (type == CUST_AICPU_KERNELS) {
        item = "aicpu kernels";
      }
      ErrorManager::GetInstance().ATCReportErrMessage("E19023", {"size", "item", "maxsize"},
        {std::to_string(size), item, std::to_string(UINT32_MAX)});
    }
    REPORT_INNER_ERROR("E19999", "Add model partition failed, partition size %zu "
                       "invalid", size);
    return PARAM_INVALID;
  }
  if (data == nullptr) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, data is null");
    REPORT_INNER_ERROR("E19999", "Add model partition failed, data is null");
    return PARAM_INVALID;
  }
  ModelPartition partition_model;
  partition_model.data = const_cast<uint8_t *>(data);
  partition_model.size = static_cast<uint32_t>(size);
  partition_model.type = type;
  if (om_file_save_helper->AddPartition(partition_model, model_index) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, partition size %zu", size);
    REPORT_CALL_ERROR("E19999", "Add model partition failed, partition size %zu", size);
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ModelHelper::SaveSizeToModelDef(const GeModelPtr &ge_model) {
  vector<int64_t> om_info;
  auto ge_model_weight = ge_model->GetWeight();
  GELOGD("SaveSizeToModelDef weight_data_size is %zu, ge_model_weight data is %p", ge_model_weight.GetSize(),
         ge_model_weight.GetData());
  om_info.push_back(ge_model_weight.GetSize());

  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGD("SaveSizeToModelDef tbe_kernels_size is %zu", tbe_kernel_store.DataSize());
  om_info.push_back(tbe_kernel_store.DataSize());

  CustAICPUKernelStore cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  GELOGD("SaveSizeToModelDef cust aicpu kernels size is %zu", cust_aicpu_kernel_store.DataSize());
  om_info.push_back(cust_aicpu_kernel_store.DataSize());

  std::shared_ptr<ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    GELOGD("SaveSizeToModelDef task_info_size is 0.");
    om_info.push_back(0);
  } else {
    size_t partition_task_size = model_task_def->ByteSizeLong();
    GELOGD("SaveSizeToModelDef task_info_size is %zu", partition_task_size);
    om_info.push_back(partition_task_size);
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(*(ge_model.get()), "om_info_list", om_info),
                   GELOGE(FAILED, "SetListInt of om_info_list failed.");
                   return FAILED);

  return SUCCESS;
}

Status ModelHelper::SaveModelDef(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                 ge::Buffer &model_buffer, size_t model_index) {
  ModelPtr model_tmp = ge::MakeShared<ge::Model>(ge_model->GetName(), ge_model->GetPlatformVersion());
  if (model_tmp == nullptr) {
    GELOGE(FAILED, "[Creat][Model]Failed, Model %s Ptr", ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Create Model %s Ptr failed", ge_model->GetName().c_str());
    return FAILED;
  }
  model_tmp->SetGraph(ge_model->GetGraph());
  model_tmp->SetVersion(ge_model->GetVersion());
  model_tmp->SetAttr(ge_model->MutableAttrMap());
  Status ret = SaveSizeToModelDef(ge_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][SizeToModelDef]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_CALL_ERROR("E19999", "Save SizeToModelDef failed, model %s, error_code %u",
                      ge_model->GetName().c_str(), ret);
    return ret;
  }

  (void)model_tmp->Save(model_buffer);
  GELOGD("MODEL_DEF size is %zu", model_buffer.GetSize());
  if (model_buffer.GetSize() > 0) {
    if (SaveModelPartition(om_file_save_helper, ModelPartitionType::MODEL_DEF, model_buffer.GetData(),
                           model_buffer.GetSize(), model_index) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, model %s, model_def size %zu, model_index %zu",
             ge_model->GetName().c_str(), model_buffer.GetSize(), model_index);
      REPORT_CALL_ERROR("E19999", "Add model graph partititon failed, model %s, model_def %zu, "
                        "model_index %zu", ge_model->GetName().c_str(), model_buffer.GetSize(), model_index);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelWeights(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     size_t model_index) {
  auto ge_model_weight = ge_model->GetWeight();
  GELOGD("WEIGHTS_DATA size is %zu, %p", ge_model_weight.GetSize(), ge_model_weight.GetData());
  // weight is not necessary
  if (ge_model_weight.GetSize() > 0) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper,
                                         ModelPartitionType::WEIGHTS_DATA,
                                         ge_model_weight.GetData(),
                                         ge_model_weight.GetSize(), model_index), "Add weight partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelTbeKernel(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeModelPtr &ge_model, size_t model_index) {
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGD("TBE_KERNELS size is %zu", tbe_kernel_store.DataSize());
  if (tbe_kernel_store.DataSize() > 0) {
    GE_CHK_STATUS_RET(
        SaveModelPartition(om_file_save_helper, ModelPartitionType::TBE_KERNELS,
                           ge_model->GetTBEKernelStore().Data(), ge_model->GetTBEKernelStore().DataSize(),
                           model_index), "Add tbe kernel partition failed");
  }
  // no need to check value, DATA->NetOutput
  (void)tbe_kernel_store.Load(tbe_kernel_store.Data(), tbe_kernel_store.DataSize());

  return SUCCESS;
}

Status ModelHelper::SaveModelCustAICPU(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeModelPtr &ge_model, size_t model_index) {
  CustAICPUKernelStore cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  GELOGD("cust aicpu kernels size is %zu", cust_aicpu_kernel_store.DataSize());
  if (cust_aicpu_kernel_store.DataSize() > 0) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper,
                                         ModelPartitionType::CUST_AICPU_KERNELS,
                                         ge_model->GetCustAICPUKernelStore().Data(),
                                         cust_aicpu_kernel_store.DataSize(), model_index),
                      "Add cust aicpu kernel partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelTaskDef(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     ge::Buffer &task_buffer, size_t model_index) {
  std::shared_ptr<ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Creat][ModelTaskDef]Failed, it is nullptr, "
           "model %s", ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Creat model task def failed, it is nullptr, model %s",
                      ge_model->GetName().c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  size_t partition_task_size = model_task_def->ByteSizeLong();
  GE_IF_BOOL_EXEC(partition_task_size == 0 || partition_task_size > INT_MAX,
                  GELOGE(FAILED, "[Check][ModelDefSize]Invalid, size %zu, model %s",
                         partition_task_size, ge_model->GetName().c_str());
                  REPORT_CALL_ERROR("E19999", "Model def size %zu check invalid, model %s",
                                    partition_task_size, ge_model->GetName().c_str());
                      return FAILED);

  task_buffer = ge::Buffer(partition_task_size);
  if (task_buffer.GetSize() == 0) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][ModelTaskDefBuffer]Failed, "
           "model def size %zu, model %s", partition_task_size, ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Allocate model task def buffer failed, model def size %zu "
                      "model %s", partition_task_size, ge_model->GetName().c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  (void)model_task_def->SerializePartialToArray(task_buffer.GetData(), static_cast<int>(partition_task_size));

  GELOGD("TASK_INFO op_size:%d, stream_num:%u", model_task_def->op().size(), model_task_def->stream_num());
  GELOGD("TASK_INFO size is %zu", partition_task_size);

  if (SaveModelPartition(om_file_save_helper, ModelPartitionType::TASK_INFO, task_buffer.GetData(),
                         partition_task_size, model_index) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Add][ModelTaskDefPartition]Failed, model def size %zu, "
           "model_index %zu, model %s",
           partition_task_size, model_index, ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Add model task def partition failed, model def size %zu "
                      "model_index %zu, model %s",
                      partition_task_size, model_index, ge_model->GetName().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelHeader(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                    size_t model_num) {
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.platform_type = ge_model->GetPlatformType();
  model_header.om_ir_version = ge_model->GetVersion();
  model_header.model_num = model_num;
  std::string platform_version = ge_model->GetPlatformVersion();

  errno_t err;
  err = memcpy_s(model_header.platform_version, PLATFORM_VERSION_LEN, platform_version.c_str(),
                 platform_version.size() + 1);
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Save][Model]Failed while allocating memory for platform_version %s, model %s, "
           "errno %d",
           platform_version.c_str(), ge_model->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999",
                      "ModelHelper save model %s failed while "
                      "allocating memory for platform_version %s, errno %d",
                      ge_model->GetName().c_str(), platform_version.c_str(), err);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  string version = reinterpret_cast<char *>(model_header.platform_version);
  GELOGD("Platform version save: %s", version.c_str());

  size_t name_size = ge_model->GetName().size();
  name_size = name_size > (MODEL_NAME_LENGTH - 1) ? (MODEL_NAME_LENGTH - 1) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, ge_model->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Save][Model]Failed while allocating memory for model %s, errno %d",
           ge_model->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999", "ModelHelper save model failed while allocating memory "
                      "for model %s,errno %d", ge_model->GetName().c_str(), err);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  string model_name = reinterpret_cast<char *>(model_header.name);
  GELOGD("Model name save:%s", model_name.c_str());
  return SUCCESS;
}

Status ModelHelper::SaveAllModelPartiton(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                         const GeModelPtr &ge_model, ge::Buffer &model_buffer, ge::Buffer &task_buffer,
                                         size_t model_index) {
  if (SaveModelDef(om_file_save_helper, ge_model, model_buffer, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelDef]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save model def failed, model %s, model index %zu",
                      ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelWeights(om_file_save_helper, ge_model, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelWeights]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save mode weights failed, model %s, model index %zu",
                      ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelTbeKernel(om_file_save_helper, ge_model, model_index) != SUCCESS) {
     GELOGE(FAILED, "[Save][ModelTbeKernel]Failed, model %s, model index %zu",
            ge_model->GetName().c_str(), model_index);
     REPORT_CALL_ERROR("E19999", "ModelHelper save model tbe kernel failed, model %s, "
                       "model index %zu", ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelCustAICPU(om_file_save_helper, ge_model, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelCustAICPU]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save model cust aicpu failed, model %s "
                      "model index %zu", ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelTaskDef(om_file_save_helper, ge_model, task_buffer, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][TaskDef]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save task def failed, model %s, model index %zu",
                      ge_model->GetName().c_str(), model_index);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::SaveToOmModel(const GeModelPtr &ge_model, const SaveParam &save_param,
                                  const std::string &output_file, ModelBufferData &model) {
  if (output_file.empty()) {
    GELOGE(FAILED, "[Save][Model]GraphBuilder SaveModel received invalid file name prefix, "
           "model %s", ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "GraphBuilder SaveModel received invalid file name prefix, "
                      "model %s", ge_model->GetName().c_str());
    return FAILED;
  }

  GE_IF_BOOL_EXEC(ge_model == nullptr, GELOGE(FAILED, "Ge_model is nullptr"); return FAILED);
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);
  ge::Buffer model_buffer;
  ge::Buffer task_buffer;

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_ATC_CMDLINE,
                   domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);
  auto ret = SaveAllModelPartiton(om_file_save_helper, ge_model, model_buffer, task_buffer);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][AllModelPartition]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save all model partition failed, model %s "
                       "error_code %u", ge_model->GetName().c_str(), ret);
    return ret;
  }

  ret = SaveModelHeader(om_file_save_helper, ge_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][ModelHeader]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save model header failed, model %s "
                       "error_code %u", ge_model->GetName().c_str(), ret);
    return ret;
  }

  ret = om_file_save_helper->SaveModel(save_param, output_file.c_str(), model, is_offline_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][Model]Failed, model %s, output file %s",
           ge_model->GetName().c_str(), output_file.c_str());
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save model failed, model %s, "
                       "output file %s", ge_model->GetName().c_str(), output_file.c_str());
    return ret;
  }
  return SUCCESS;
}

Status ModelHelper::SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const SaveParam &save_param,
                                      const std::string &output_file, ModelBufferData &model, bool is_unknown_shape) {
  GE_CHECK_NOTNULL(ge_root_model);
  GE_IF_BOOL_EXEC(ge_root_model == nullptr,
                  GELOGE(FAILED, "[Check][GERootModel]Ge_root_model is nullptr");
                  REPORT_INNER_ERROR("E19999", "Ge_root_model check failed, it is nullptr");
                  return FAILED);

  auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_IF_BOOL_EXEC(name_to_ge_model.empty(),
                  GELOGE(FAILED, "[Get][SubModel]Ge_root_model has no sub model");
                  REPORT_INNER_ERROR("E19999", "Ge_root_model has no sub model");
                  return FAILED);
  GE_IF_BOOL_EXEC(output_file.empty(),
                  GELOGE(FAILED, "[Save][Model]GraphBuilder SaveModel received invalid "
                         "file name prefix");
                  REPORT_INNER_ERROR("E19999", "GraphBuilder SaveModel received invalid "
                                     "file name prefix");
                  return FAILED);
  if (!is_unknown_shape) {
    auto &model_root = name_to_ge_model.begin()->second;
    GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(model_root.get()), ATTR_MODEL_ATC_CMDLINE,
                     domi::GetContext().atc_cmdline),
                     GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                     return FAILED);
    return SaveToOmModel(model_root, save_param, output_file, model);
  }

  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);

  auto &first_ge_model = name_to_ge_model.at(ge_root_model->GetRootGraph()->GetName());
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(first_ge_model.get()), ATTR_MODEL_ATC_CMDLINE,
                   domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);

  // ge root model must be the first to be loaded
  vector<string> model_names{ge_root_model->GetRootGraph()->GetName()};
  for (auto &item : name_to_ge_model) {
    if (item.first != model_names.front()) {
      model_names.emplace_back(item.first);
    }
  }

  vector<ge::Buffer> model_buffers(model_names.size());
  vector<ge::Buffer> task_buffers(model_names.size());

  size_t cur_index = 0;

  if (model_names.size() > 1) {
    GELOGD("only save first model MODEL_DEF");
    if (SaveModelDef(om_file_save_helper, first_ge_model, model_buffers[cur_index], cur_index) != SUCCESS) {
      GELOGE(FAILED, "[Save][ModelDef]Failed, cur_index %zu", cur_index);
      REPORT_INNER_ERROR("E19999", "Save model def failed, cur_index %zu", cur_index);
      return FAILED;
    }
    ++cur_index;
  }

  for (; cur_index < model_names.size(); ++cur_index) {
    auto model_name = model_names[cur_index];
    GELOGD("cur model %s index is %zu", model_name.c_str(), cur_index);
    const GeModelPtr &ge_model = name_to_ge_model.at(model_name);
    auto ret = SaveAllModelPartiton(om_file_save_helper, ge_model, model_buffers[cur_index],
                                    task_buffers[cur_index], cur_index);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Save][AllModelPartition]Failed, model name %s, cur_index %zu",
             model_name.c_str(), cur_index);
      REPORT_CALL_ERROR("E19999", "Save all model %s partition failed, cur_index %zu",
                        model_name.c_str(), cur_index);
      return INTERNAL_ERROR;
    }
  }

  auto ret = SaveModelHeader(om_file_save_helper, first_ge_model, model_names.size());
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Save][ModelHeader]Failed, model name %s",
           first_ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Save model %s header failed", first_ge_model->GetName().c_str());
    return INTERNAL_ERROR;
  }

  ret = om_file_save_helper->SaveRootModel(save_param, output_file.c_str(), model, is_offline_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][Model]OmFileSaveHelper save model eturn fail, output_file %s",
           output_file.c_str());
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save model return fail, output_file %s",
                      output_file.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file) {
  if (output_file.empty()) {
    GELOGE(FAILED, "[Save][Model]Received invalid file name prefix, output_file %s", output_file.c_str());
    REPORT_INNER_ERROR("E19999", "Save model received invalid file name prefix, output_file %s", output_file.c_str());
    return FAILED;
  }
  // Get computegraph from graph
  auto compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(FAILED, "[Save][Model]Failed for compute_graph null");
    REPORT_INNER_ERROR("E19999", "Save model failed for compute_graph null");
    return FAILED;
  }
  GE_DUMP(compute_graph, "OriginalGraph");
  // Model
  ModelPtr model_ptr = ge::MakeShared<ge::Model>();
  GE_CHECK_NOTNULL_EXEC(model_ptr, return MEMALLOC_FAILED);
  std::string original_model_name = compute_graph->GetName() + "_original";
  model_ptr->SetName(original_model_name);
  model_ptr->SetGraph(graph);
  model_ptr->SetVersion(static_cast<uint32_t>(OM_PROTO_VERSION));
  string framework_version;
  Status frame_rt = PlatformVersionManager::GetPlatformVersion(framework_version);
  if (frame_rt == SUCCESS) {
    uint32_t counter = 0;
    string model_framework_version = framework_version + "." + std::to_string(counter);
    model_ptr->SetPlatformVersion(model_framework_version);
  }
  // Model def
  ge::Buffer model_buffer;
  ge::graphStatus status = model_ptr->Save(model_buffer);
  if (status != ge::GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Save][Model]Failed for save buffer fail, model %s",
           model_ptr->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Save model %s failed for save buffer fail",
                      model_ptr->GetName().c_str());
    return FAILED;
  }
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL_EXEC(om_file_save_helper, return MEMALLOC_FAILED);
  ModelPartition partition_model;
  partition_model.data = model_buffer.GetData();
  partition_model.size = static_cast<uint32_t>(model_buffer.GetSize());
  partition_model.type = ModelPartitionType::MODEL_DEF;
  GELOGI("Original Model type[%u],size[%u]", partition_model.type, partition_model.size);
  if (partition_model.data != nullptr && partition_model.size > 0) {
    (void)om_file_save_helper->AddPartition(partition_model);
    // Condition of AddPartition is established, no need to check value
  }
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.om_ir_version = model_ptr->GetVersion();
  model_header.headsize = MODEL_FILE_HEAD_LEN;
  std::string platform_version = model_ptr->GetPlatformVersion();
  errno_t err = memcpy_s(model_header.platform_version, PLATFORM_VERSION_LEN, platform_version.c_str(),
                         platform_version.size() + 1);
  if (err != EOK) {
    GELOGE(FAILED, "[Save][Model]Failed for platform_version %s, model %s, errno %d",
           platform_version.c_str(), model_ptr->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999", "Save model %s failed for platform_version %s, errno %d",
                      model_ptr->GetName().c_str(), platform_version.c_str(), err);
    return FAILED;
  }
  size_t name_size = model_ptr->GetName().size();
  name_size = name_size > (MODEL_NAME_LENGTH - 1) ? (MODEL_NAME_LENGTH - 1) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, model_ptr->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(FAILED, "[Save][Model]Failed for memory copy %s failed, errno %d",
           model_ptr->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999", "Save model failed for memory copy %s failed, errno %d",
                      model_ptr->GetName().c_str(), err);
    return FAILED;
  }
  ModelBufferData model;
  Status ret = om_file_save_helper->SaveModelToFile(output_file.c_str(), model, is_offline_);
  return (ret == SUCCESS ? SUCCESS : FAILED);
}

Status ModelHelper::LoadModel(const ge::ModelData &model_data) {
  if (model_data.model_data == nullptr || model_data.model_len == 0) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Load][Model]Model_data is nullptr or model_data_size is 0");
    REPORT_INNER_ERROR("E19999", "Load model failed, "
                       "Model_data is nullptr or model_data_size is 0");
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  if (is_assign_model_) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "[Load][Model]Model helper has already loaded!");
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED;
  }

  if (ReleaseLocalModelData() != SUCCESS) {
    GELOGE(ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA, "[Release][ModelData]Failed.");
    return ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA;
  }

  Status status = ModelParserBase::ParseModelContent(model_data, model_addr_tmp_, model_len_tmp_);
  if (status != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Parse][ModelContent]Failed!");
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  file_header_ = reinterpret_cast<ModelFileHeader *>(model_data.model_data);
  OmFileLoadHelper om_load_helper;
  status = om_load_helper.Init(model_addr_tmp_, model_len_tmp_);
  if (status != SUCCESS) {
    GELOGE(status, "[Init][OmLoadHelper]Failed");
    model_addr_tmp_ = nullptr;
    return status;
  }
  auto partition_table = reinterpret_cast<ModelPartitionTable *>(model_addr_tmp_);
  if (partition_table->num == kOriginalOmPartitionNum) {
    model_addr_tmp_ = nullptr;
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][OmModel]Error, please use executable om model");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  // Encrypt model need to del temp model/no encrypt model don't need to del model
  model_addr_tmp_ = nullptr;

  status = GenerateGeModel(om_load_helper);
  if (status != SUCCESS) {
    GELOGE(status, "[Generate][GEModel]Failed");
    return status;
  }
  GELOGD("in ModelHelper::LoadModel, is_assign_model_ is setted to true!");
  is_assign_model_ = true;
  return SUCCESS;
}

Status ModelHelper::LoadRootModel(const ge::ModelData &model_data) {
  if (model_data.model_data == nullptr || model_data.model_len == 0) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "[Load][RootModel] "
           "Model_data is nullptr or model data is empty.");
    REPORT_INNER_ERROR("E19999", "Load root model failed, model_data is nullptr or its size is 0");
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  if (is_assign_model_) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED,
           "[Load][RootModel]Model helper has already loaded!");
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED;
  }

  if (ReleaseLocalModelData() != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Release][ModelData]Failed.");
    return INTERNAL_ERROR;
  }

  Status status = ModelParserBase::ParseModelContent(model_data, model_addr_tmp_, model_len_tmp_);
  if (status != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Parse][RootModelContent]Failed!");
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  file_header_ = reinterpret_cast<ModelFileHeader *>(model_data.model_data);

  //model verison 1.0 file header does not have model_num member
  is_unknown_shape_model_ = file_header_->version >= ge::MODEL_VERSION &&
                            file_header_->model_num > kStatiOmFileModelNum;
  GELOGD("Cur om model is ge root model or no %d, model version %u", is_unknown_shape_model_, file_header_->version);

  OmFileLoadHelper om_load_helper;
  if (is_unknown_shape_model_) {
    auto model_num = file_header_->model_num;
    status = om_load_helper.Init(model_addr_tmp_, model_len_tmp_, model_num);
  } else {
    status = om_load_helper.Init(model_addr_tmp_, model_len_tmp_);
  }
  if (status != SUCCESS) {
    GELOGE(status, "[Init][OmLoadHelper]Failed");
    model_addr_tmp_ = nullptr;
    return status;
  }
  // Encrypt model need to del temp model/no encrypt model don't need to del model
  model_addr_tmp_ = nullptr;

  status = GenerateGeRootModel(om_load_helper);
  if (status != SUCCESS) {
    GELOGE(status, "[Generate][GERootModel]Failed");
    return status;
  }
  GELOGD("In ModelHelper::LoadRootModel, is_assign_model_ is setted to true!");
  is_assign_model_ = true;
  return SUCCESS;
}

Status ModelHelper::GenerateGeModel(OmFileLoadHelper &om_load_helper) {
  model_ = ge::MakeShared<ge::GeModel>();
  GE_CHECK_NOTNULL(model_);
  Status ret = LoadModelData(om_load_helper);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED;
  }
  ret = LoadWeights(om_load_helper);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED;
  }
  ret = LoadTask(om_load_helper);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
  }
  ret = LoadTBEKernelStore(om_load_helper);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
  }
  ret = LoadCustAICPUKernelStore(om_load_helper);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::GenerateGeRootModel(OmFileLoadHelper &om_load_helper) {
  GELOGD("Begin to generate ge root model");
  root_model_ = ge::MakeShared<ge::GeRootModel>();
  GE_CHECK_NOTNULL(root_model_);
  if (!is_unknown_shape_model_) {
    if (GenerateGeModel(om_load_helper) != SUCCESS) {
      GELOGE(FAILED, "[Generate][GERootModel]Failed");
      REPORT_CALL_ERROR("E19999", "Generate GE root model failed");
      return FAILED;
    }
    GE_CHECK_NOTNULL(model_);
    root_model_->SetRootGraph(GraphUtils::GetComputeGraph(model_->GetGraph()));
    return SUCCESS;
  }

  bool is_first_model = true;
  for (size_t mode_index = 0;  mode_index < file_header_->model_num; ++mode_index) {
    GeModelPtr cur_model = ge::MakeShared<ge::GeModel>();
    Status ret = LoadModelData(om_load_helper, cur_model, mode_index);
    if (ret != SUCCESS) {
      return ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED;
    }

    if (is_first_model) {
      is_first_model = false;
      root_model_->SetRootGraph(GraphUtils::GetComputeGraph(cur_model->GetGraph()));
      root_model_->SetModelId(cur_model->GetModelId());
      root_model_->SetModelName(cur_model->GetName());
      model_ = cur_model;
      continue;
    }

    ret = LoadWeights(om_load_helper, cur_model, mode_index);
    if (ret != SUCCESS) {
      return ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED;
    }

    ret = LoadTBEKernelStore(om_load_helper, cur_model, mode_index);
    if (ret != SUCCESS) {
      return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
    }

    ret = LoadCustAICPUKernelStore(om_load_helper, cur_model, mode_index);
    if (ret != SUCCESS) {
      return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
    }

    ret = LoadTask(om_load_helper, cur_model, mode_index);
    if (ret != SUCCESS) {
      return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
    }
    root_model_->SetSubgraphInstanceNameToModel(cur_model->GetName(), cur_model);
  }

  return SUCCESS;
}

Status ModelHelper::LoadModelData(OmFileLoadHelper &om_load_helper) {
  ModelPartition partition_model_def;
  // no need to check value, DATA->NetOutput
  om_load_helper.GetModelPartition(ModelPartitionType::MODEL_DEF, partition_model_def);
  GELOGD("Model_def partition addr:%p,size:%u", partition_model_def.data, partition_model_def.size);

  ge::Model model;
  if (ge::Model::Load(partition_model_def.data, partition_model_def.size, model) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Load][Model]Failed, model_def partition addr:%p, size:%u",
           partition_model_def.data, partition_model_def.size);
    REPORT_CALL_ERROR("E19999", "Load model failed, model_def partition addr:%p, size:%u",
                      partition_model_def.data, partition_model_def.size);
    return INTERNAL_ERROR;
  }

  SetModelToGeModel(model_, model);
  return SUCCESS;
}

void ModelHelper::SetModelToGeModel(GeModelPtr &ge_model, Model &model) {
  ge_model->SetGraph(model.GetGraph());
  ge_model->SetName(model.GetName());
  ge_model->SetVersion(model.GetVersion());
  ge_model->SetPlatformVersion(model.GetPlatformVersion());
  ge_model->SetAttr(model.MutableAttrMap());
}

Status ModelHelper::LoadModelData(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index) {
  ModelPartition partition_model_def;
  // no need to check value, DATA->NetOutput
  om_load_helper.GetModelPartition(ModelPartitionType::MODEL_DEF, partition_model_def, mode_index);
  GELOGD("Model_def partition addr:%p,size:%u", partition_model_def.data, partition_model_def.size);

  ge::Model model;
  if (ge::Model::Load(partition_model_def.data, partition_model_def.size, model) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Load][Model]Failed, model_def partition addr:%p, size:%u",
           partition_model_def.data, partition_model_def.size);
    REPORT_CALL_ERROR("E19999", "Load model failed, model_def partition addr:%p, size:%u",
                      partition_model_def.data, partition_model_def.size);
    return INTERNAL_ERROR;
  }

  SetModelToGeModel(cur_model, model);
  return SUCCESS;
}

Status ModelHelper::LoadWeights(OmFileLoadHelper &om_load_helper) {
  ModelPartition partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition) != SUCCESS) {
    GELOGE(FAILED, "[Get][ModelWeightPartition]Failed, GetWeight size:%u", partition.size);
    REPORT_CALL_ERROR("E19999", "[Get][ModelPartition]Failed, GetWeight size:%u",
                      partition.size);
    return FAILED;
  }
  ge::Buffer weight = ge::Buffer::CopyFrom(partition.data, partition.size);
  model_->SetWeight(weight);

  GELOGD("GetWeight size:%u", partition.size);
  return SUCCESS;
}

Status ModelHelper::LoadWeights(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index) {
  ModelPartition partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition, mode_index) != SUCCESS) {
    GELOGE(FAILED, "[Get][ModelPartition]Failed, GetWeight size:%u", partition.size);
    REPORT_CALL_ERROR("E19999", "[Get][ModelPartition]Failed, GetWeight size:%u",
                      partition.size);
    return FAILED;
  }
  ge::Buffer weight = ge::Buffer::CopyFrom(partition.data, partition.size);
  cur_model->SetWeight(weight);

  GELOGD("GetWeight size:%u", partition.size);
  return SUCCESS;
}

Status ModelHelper::LoadTask(OmFileLoadHelper &om_load_helper) {
  ModelPartition task_partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition) != SUCCESS) {
    GELOGE(FAILED, "[Get][ModelTaskPartition]Failed, task_partition size:%u", task_partition.size);
    REPORT_CALL_ERROR("E19999", "Get model task partition failed, task_partition size %u",
                      task_partition.size);
    return FAILED;
  }
  std::shared_ptr<ModelTaskDef> task = ge::MakeShared<ModelTaskDef>();
  GE_CHECK_NOTNULL(task);
  if (task_partition.size != 0) {
    if (!ReadProtoFromArray(task_partition.data, task_partition.size, task.get())) {
      GELOGE(INTERNAL_ERROR, "[Read][ProtoFromArray]Failed, task_partition size %u",
             task_partition.size);
      REPORT_CALL_ERROR("E19999", "Read proto from array failed, task_partition size %u",
                        task_partition.size);
      return INTERNAL_ERROR;
    }
    GELOGD("TASK_INFO op_size:%d, stream_num:%u", task->op().size(), task->stream_num());
  }
  model_->SetModelTaskDef(task);
  return SUCCESS;
}

Status ModelHelper::LoadTask(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index) {
  ModelPartition task_partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition, mode_index) != SUCCESS) {
    GELOGE(FAILED, "Get task model partition failed.");
    GELOGE(FAILED, "[Get][ModelTaskPartition]Failed, task_partition size %u, mode_index %zu",
           task_partition.size, mode_index);
    REPORT_CALL_ERROR("E19999", "Get model task partition failed, "
                       "task_partition size %u, mode_index %zu", task_partition.size, mode_index);
    return FAILED;
  }
  std::shared_ptr<ModelTaskDef> task = ge::MakeShared<ModelTaskDef>();
  GE_CHECK_NOTNULL(task);
  if (task_partition.size != 0) {
    if (!ReadProtoFromArray(task_partition.data, task_partition.size, task.get())) {
      GELOGE(INTERNAL_ERROR, "[Read][ProtoFromArray]Failed, task_partition size %u",
             task_partition.size);
      REPORT_CALL_ERROR("E19999", "Read proto from array failed, task_partition size %u",
                        task_partition.size);
      return INTERNAL_ERROR;
    }
    GELOGD("TASK_INFO op_size:%d, stream_num:%u", task->op().size(), task->stream_num());
  }
  cur_model->SetModelTaskDef(task);
  return SUCCESS;
}

Status ModelHelper::LoadTBEKernelStore(OmFileLoadHelper &om_load_helper) {
  // Load tbe kernels
  ModelPartition partition_kernel_def;
  TBEKernelStore kernel_store;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TBE_KERNELS, partition_kernel_def) == SUCCESS) {
    GELOGD("Kernels partition size:%u", partition_kernel_def.size);
    if (kernel_store.Load(partition_kernel_def.data, partition_kernel_def.size)) {
      GELOGD("Load tbe kernels success");
    } else {
      GELOGW("Load tbe kernels failed");
    }
  }
  model_->SetTBEKernelStore(kernel_store);
  return SUCCESS;
}

Status ModelHelper::LoadTBEKernelStore(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index) {
  // Load tbe kernels
  ModelPartition partition_kernel_def;
  TBEKernelStore kernel_store;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TBE_KERNELS, partition_kernel_def, mode_index) ==
      SUCCESS) {
    GELOGD("Kernels partition size:%u", partition_kernel_def.size);
    if (kernel_store.Load(partition_kernel_def.data, partition_kernel_def.size)) {
      GELOGD("Load tbe kernels success");
    } else {
      GELOGW("Load tbe kernels failed");
    }
  }
  cur_model->SetTBEKernelStore(kernel_store);
  return SUCCESS;
}

Status ModelHelper::LoadCustAICPUKernelStore(OmFileLoadHelper &om_load_helper) {
  // Load cust aicpu kernels
  ModelPartition partition_kernel_def;
  CustAICPUKernelStore kernel_store;
  if (om_load_helper.GetModelPartition(ModelPartitionType::CUST_AICPU_KERNELS, partition_kernel_def) == SUCCESS) {
    GELOGD("Kernels partition size:%u", partition_kernel_def.size);
    if (kernel_store.Load(partition_kernel_def.data, partition_kernel_def.size)) {
      GELOGD("Load cust aicpu kernels success");
    } else {
      GELOGW("Load cust aicpu kernels failed");
    }
  }
  model_->SetCustAICPUKernelStore(kernel_store);
  return SUCCESS;
}

Status ModelHelper::LoadCustAICPUKernelStore(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model,
                                             size_t mode_index) {
  // Load cust aicpu kernels
  ModelPartition partition_kernel_def;
  CustAICPUKernelStore kernel_store;
  if (om_load_helper.GetModelPartition(ModelPartitionType::CUST_AICPU_KERNELS, partition_kernel_def, mode_index)
      == SUCCESS) {
    GELOGD("Kernels partition size:%u", partition_kernel_def.size);
    if (kernel_store.Load(partition_kernel_def.data, partition_kernel_def.size)) {
      GELOGD("Load cust aicpu kernels success");
    } else {
      GELOGW("Load cust aicpu kernels failed");
    }
  }
  cur_model->SetCustAICPUKernelStore(kernel_store);
  return SUCCESS;
}

GeModelPtr ModelHelper::GetGeModel() {
  if (model_ != nullptr) {
    return model_;
  }

  GELOGD("Model has not been loaded!");
  std::shared_ptr<ge::GeModel> out_model = ge::MakeShared<ge::GeModel>();
  if (out_model == nullptr) {
    return nullptr;
  }
  return out_model;
}

GeRootModelPtr ModelHelper::GetGeRootModel() {
  if (root_model_ != nullptr) {
    return root_model_;
  }

  GELOGD("Model has not been loaded!");
  std::shared_ptr<ge::GeRootModel> out_model = ge::MakeShared<ge::GeRootModel>();
  if (out_model == nullptr) {
    return nullptr;
  }
  return out_model;
}

Status ModelHelper::ReleaseLocalModelData() noexcept {
  Status result = SUCCESS;
  if (model_addr_tmp_ != nullptr) {
    errno_t ret = memset_s(static_cast<void *>(model_addr_tmp_), model_len_tmp_, 0, model_len_tmp_);
    if (ret != EOK) {
      GELOGE(FAILED, "[Memset][Memory]Failed, error-code %d", ret);
      REPORT_CALL_ERROR("E19999", "Failed to memset memory, error-code %d", ret);
      result = FAILED;
    }
    delete[] model_addr_tmp_;
    model_addr_tmp_ = nullptr;
    model_len_tmp_ = 0;
  }
  return result;
}

Status ModelHelper::GetBaseNameFromFileName(const string &file_name, string &base_name) {
  GELOGD("Get base_name from file, file_name:%s", file_name.c_str());
  GE_CHK_BOOL_EXEC_WARN(!file_name.empty(), return FAILED, "File path may not valid, check params --output");
  size_t start_position = 0;
  // using output as base_name (ignore ".om")
  size_t filename_suffixes = 3;
  if (file_name.find_last_of('/') != string::npos) {
    start_position = file_name.find_last_of('/') + 1;
  }
  size_t end_position = file_name.length() - filename_suffixes;
  base_name = file_name.substr(start_position, end_position - start_position);
  GE_CHK_BOOL_EXEC_WARN(!base_name.empty(), return FAILED, "Get base_name failed, check params --output");
  return SUCCESS;
}

Status ModelHelper::GetModelNameFromMergedGraphName(const string &graph_name, string &model_name) {
  GELOGD("Get model_name from graph_name, graph_name:%s", graph_name.c_str());
  // this can only be used after merged graph(graph name will be append with "_x", x is index);
  GE_CHK_BOOL_EXEC_WARN(!graph_name.empty(), return FAILED, "File path may not valid, check params --output");
  size_t start_position = 0;
  size_t end_position = graph_name.length();
  // using graph as model_name (ignore "_x", x is the index of graph)
  if (graph_name.find_last_of('_') != string::npos) {
    end_position = graph_name.find_last_of('_');
  }
  model_name = graph_name.substr(start_position, end_position);
  GE_CHK_BOOL_EXEC_WARN(!model_name.empty(), return FAILED, "Get model_name failed, check params --output");
  return SUCCESS;
}

Status ModelTool::GetModelInfoFromOm(const char *model_file, ge::proto::ModelDef &model_def, uint32_t &modeldef_size) {
  GE_CHECK_NOTNULL(model_file);
  ge::ModelData model;
  int32_t priority = 0;

  Status ret = ModelParserBase::LoadFromFile(model_file, priority, model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][ModelInfo]Failed from file %s, error_code %u", model_file, ret);
    REPORT_CALL_ERROR("E19999", "Load model info failed from file %s, error_code %u",
                      model_file, ret);
    return ret;
  }
  std::function<void()> callback = [&]() {
    if (model.model_data != nullptr) {
      delete[] reinterpret_cast<char *>(model.model_data);
      model.model_data = nullptr;
    }
  };
  GE_MAKE_GUARD(release, callback);

  uint8_t *model_data = nullptr;
  uint32_t model_len = 0;
  ret = ModelParserBase::ParseModelContent(model, model_data, model_len);
  if (ret != SUCCESS) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10003",
      {"parameter", "value", "reason"}, {"om", model_file, "invalid om file, can't be parsed"});
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Parse][ModelContent]Failed because of invalid om file %s, please check om param", model_file);
    return ret;
  }

  OmFileLoadHelper om_load_helper;
  ret = om_load_helper.Init(model_data, model_len);
  if (ret != SUCCESS) {
    REPORT_INNER_ERROR("E19999", "Init om file %s failed", model_file);
    GELOGE(ge::FAILED, "[Init][OmFile]Failed, model_file %s", model_file);
    return ret;
  }

  ModelPartition ir_part;
  ret = om_load_helper.GetModelPartition(MODEL_DEF, ir_part);
  if (ret != SUCCESS) {
    REPORT_INNER_ERROR("E19999", "Get model partition failed, model_file %s", model_file);
    GELOGE(ge::FAILED, "[Get][ModelPart]Failed, model_file %s", model_file);
    return ret;
  }

  bool flag = ReadProtoFromArray(ir_part.data, ir_part.size, &model_def);
  if (!flag) {
    ret = INTERNAL_ERROR;
    REPORT_INNER_ERROR("E19999", "Read proto from array failed, model_file %s", model_file);
    GELOGE(ret, "[Read][ProtoFromArray]Failed, model_file %s", model_file);
    return ret;
  }
  modeldef_size = ir_part.size;
  return ret;
}

Status ModelTool::GetModelInfoFromPbtxt(const char *model_file, ge::proto::ModelDef &model_def) {
  GE_CHECK_NOTNULL(model_file);
  ge::ModelData model;
  int32_t priority = 0;

  Status ret = ModelParserBase::LoadFromFile(model_file, priority, model);
  auto free_model_data = [](void **ptr) -> void {
    if (ptr != nullptr && *ptr != nullptr) {
      delete[] reinterpret_cast<char *>(*ptr);
      *ptr = nullptr;
    }
  };
  if (ret != SUCCESS) {
    free_model_data(&model.model_data);
    GELOGE(ret, "[Load][ModelInfo]Failed from file %s, error_code %u", model_file, ret);
    REPORT_CALL_ERROR("E19999", "Load model info failed from file %s, error_code %u",
                      model_file, ret);
    return ret;
  }

  try {
    bool flag = google::protobuf::TextFormat::ParseFromString(reinterpret_cast<char *>(model.model_data), &model_def);
    if (!flag) {
      free_model_data(&model.model_data);
      REPORT_INNER_ERROR("E19999", "Parse model info failed from string, model_file %s", model_file);
      GELOGE(FAILED, "[Parse][ModelInfo]Failed from string, model_file %s", model_file);
      return FAILED;
    }
    free_model_data(&model.model_data);
    return SUCCESS;
  } catch (google::protobuf::FatalException &e) {
    free_model_data(&model.model_data);
    REPORT_INNER_ERROR("E19999", "Parse model info failed from string, exception message %s, model_file %s",
                       e.what(), model_file);
    GELOGE(FAILED, "[Parse][ModelInfo]Failed from string, exception message %s, model_file %s",
           e.what(), model_file);
    return FAILED;
  }
}
}  // namespace ge
