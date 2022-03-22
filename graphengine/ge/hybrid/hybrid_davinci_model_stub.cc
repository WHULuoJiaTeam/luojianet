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

#include "hybrid/hybrid_davinci_model.h"

namespace ge {
namespace hybrid {
HybridDavinciModel::~HybridDavinciModel() {}

std::unique_ptr<HybridDavinciModel> HybridDavinciModel::Create(const GeRootModelPtr &ge_root_model) {
  return std::unique_ptr<HybridDavinciModel>(new (std::nothrow)HybridDavinciModel());
}

Status HybridDavinciModel::Init() {
  return UNSUPPORTED;
}

Status HybridDavinciModel::Execute(const std::vector<DataBuffer> &inputs,
                                   const std::vector<GeTensorDesc> &input_desc,
                                   std::vector<DataBuffer> &outputs,
                                   std::vector<GeTensorDesc> &output_desc,
                                   rtStream_t stream) {
  return UNSUPPORTED;
}

Status HybridDavinciModel::Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
  return UNSUPPORTED;
}

Status HybridDavinciModel::ModelRunStart() {
  return UNSUPPORTED;
}

Status HybridDavinciModel::ModelRunStop() {
  return UNSUPPORTED;
}

Status HybridDavinciModel::EnqueueData(const shared_ptr<InputDataWrapper> &data) {
  return UNSUPPORTED;
}

void HybridDavinciModel::SetListener(const shared_ptr<ModelListener> &listener) {
}

void HybridDavinciModel::SetModelId(uint32_t model_id) {
}

void HybridDavinciModel::SetDeviceId(uint32_t device_id) {
}

void HybridDavinciModel::SetOmName(const string &om_name) {
}

uint64_t HybridDavinciModel::GetSessionId() {
  return 0;
}

uint32_t HybridDavinciModel::GetDataInputerSize() {
  return 0;
}

uint32_t HybridDavinciModel::GetDeviceId() const {
  return 0;
}

Status HybridDavinciModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) {
  return UNSUPPORTED;
}

void HybridDavinciModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) {
}

void HybridDavinciModel::GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) {
}

Status HybridDavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                                  vector<InputOutputDescInfo> &output_desc,
                                                  std::vector<uint32_t> &input_formats,
                                                  std::vector<uint32_t> &output_formats) {
  return UNSUPPORTED;
}

void HybridDavinciModel::SetModelDescVersion(bool is_new_model_desc) {
}

bool HybridDavinciModel::GetRunningFlag() const {
  return false;
}

Status HybridDavinciModel::SetRunAsyncListenerCallback(const RunAsyncCallback &callback) {
  return UNSUPPORTED;
}

bool HybridDavinciModel::GetOpDescInfo(uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info) const {
  return true;
}
Status HybridDavinciModel::GetOpAttr(const std::string &op_name, const std::string &attr_name,
                                     std::string &attr_value) const {
  return UNSUPPORTED;
}
}  // namespace hybrid
}  // namespace ge