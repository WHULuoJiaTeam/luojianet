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

#include "common/model/ge_model.h"
#include <utility>
#include "framework/common/debug/log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace ge {
void GeModel::Init() {
  (void)AttrUtils::SetInt(this, ATTR_MODEL_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_P2P_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_STREAM_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_EVENT_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_LABEL_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_WEIGHT_SIZE, 0);
  (void)AttrUtils::SetStr(this, ATTR_MODEL_TARGET_TYPE, TARGET_TYPE_MINI);
  version_ = 0;
  // default attrSize = 5
}

GeModel::GeModel() {
  Init();
}

const Graph &GeModel::GetGraph() const { return this->graph_; }

std::shared_ptr<domi::ModelTaskDef> GeModel::GetModelTaskDefPtr() const { return this->task_; }

const TBEKernelStore &GeModel::GetTBEKernelStore() const { return this->tbe_kernal_store_; }

const CustAICPUKernelStore &GeModel::GetCustAICPUKernelStore() const { return this->cust_aicpu_kernal_store_; }

Buffer GeModel::GetWeight() const { return this->weights_buffer_; }

std::string GeModel::GetName() const { return this->name_; }

uint32_t GeModel::GetVersion() const { return this->version_; }

std::string GeModel::GetPlatformVersion() const { return this->platform_version_; }

uint8_t GeModel::GetPlatformType() const { return this->platform_type_; }

void GeModel::SetGraph(const Graph &graph) { this->graph_ = graph; }

void GeModel::SetModelTaskDef(const std::shared_ptr<domi::ModelTaskDef> &task) { this->task_ = task; }

void GeModel::SetTBEKernelStore(const TBEKernelStore &tbe_kernal_store) {
  this->tbe_kernal_store_ = tbe_kernal_store;
}

void GeModel::SetCustAICPUKernelStore(const CustAICPUKernelStore &cust_aicpu_kernal_store) {
  this->cust_aicpu_kernal_store_ = cust_aicpu_kernal_store;
}

void GeModel::SetWeight(const Buffer &weights_buffer) { this->weights_buffer_ = weights_buffer; }

void GeModel::SetName(const std::string &name) { this->name_ = name; }

void GeModel::SetVersion(uint32_t version) { this->version_ = version; }

void GeModel::SetPlatformVersion(const std::string &platform_version) { this->platform_version_ = platform_version; }

void GeModel::SetPlatformType(uint8_t platform_type) { this->platform_type_ = platform_type; }

void GeModel::SetAttr(const ProtoAttrMap &attrs) { attrs_ = attrs; }

ProtoAttrMap &GeModel::MutableAttrMap() { return attrs_; }

ConstProtoAttrMap &GeModel::GetAttrMap() const {
  return attrs_;
}

Status GeModel::GetSessionId(uint32_t model_id, uint64_t &session_id) const {
  auto it = model_id_to_session_id_map_.find(model_id);
  if (it != model_id_to_session_id_map_.end()) {
    session_id = it->second;
    return SUCCESS;
  }
  GELOGW("No session id were found with model id [%u].", model_id);
  return INTERNAL_ERROR;
}
}  // namespace ge
