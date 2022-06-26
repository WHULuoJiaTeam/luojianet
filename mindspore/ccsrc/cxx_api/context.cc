/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "include/api/context.h"
#include <any>
#include <map>
#include <type_traits>
#include "cxx_api/factory.h"
#include "utils/log_adapter.h"

constexpr auto kModelOptionCpuEnableFP16 = "mindspore.option.cpu.enable_fp16";
constexpr auto kModelOptionGPUEnableFP16 = "mindspore.option.gpu.enable_fp16";
constexpr auto kModelOptionKirinNpuFrequency = "mindspore.option.kirin_npu.frequency";
constexpr auto kModelOptionDeviceID = "mindspore.option.device_id";
constexpr auto kModelOptionGPUDeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionGPUPrecisionMode = "mindspore.option.gpu.precision_mode";
constexpr auto kModelOptionAscend910DeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionAscend310DeviceID = kModelOptionDeviceID;
constexpr auto kModelOptionAscend310InsertOpCfgPath = "mindspore.option.ascend310.insert_op_config_file_path";
constexpr auto kModelOptionAscend310InputFormat = "mindspore.option.ascend310.input_format";
constexpr auto kModelOptionAscend310InputShapeMap = "mindspore.option.ascend310.input_shape_map";
constexpr auto kModelOptionAscend310InputShape = "mindspore.option.ascend310.input_shape";
constexpr auto kModelOptionAscend310OutputType = "mindspore.option.ascend310.output_type";
constexpr auto kModelOptionAscend310PrecisionMode = "mindspore.option.ascend310.precision_mode";
constexpr auto kModelOptionAscend310OpSelectImplMode = "mindspore.option.ascend310.op_select_impl_mode";
constexpr auto KModelOptionAscend310FusionSwitchCfgPath = "mindspore.option.ascend310.fusion_switch_config_file_path";
constexpr auto kModelOptionAscend310DynamicBatchSize = "mindspore.option.ascend310.dynamic_batch_size";
constexpr auto kModelOptionAscend310BufferOptimize = "mindspore.option.ascend310.buffer_optimize";

namespace mindspore {
class Allocator {};

struct Context::Data {
  std::vector<std::shared_ptr<DeviceInfoContext>> device_info_list;
  int32_t thread_num;
  bool enable_parallel_ = false;
  std::vector<int32_t> affinity_core_list_;
  int affinity_mode_ = 2;
};

struct DeviceInfoContext::Data {
  std::map<std::string, std::any> params;
};

Context::Context() : data_(std::make_shared<Data>()) {}

template <class T, typename U = std::remove_cv_t<std::remove_reference_t<T>>>
static const U &GetValue(const std::shared_ptr<DeviceInfoContext::Data> &data, const std::string &key) {
  static const U empty_result{};
  if (data == nullptr) {
    return empty_result;
  }
  auto iter = data->params.find(key);
  if (iter == data->params.end()) {
    return empty_result;
  }
  const std::any &value = iter->second;
  if (value.type() != typeid(U)) {
    return empty_result;
  }

  return std::any_cast<const U &>(value);
}

void Context::SetThreadNum(int32_t thread_num) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->thread_num = thread_num;
}
int32_t Context::GetThreadNum() const {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->thread_num;
}

void Context::SetEnableParallel(bool is_parallel) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->enable_parallel_ = is_parallel;
}

bool Context::GetEnableParallel() const {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->enable_parallel_;
}

void Context::SetThreadAffinity(int mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->affinity_mode_ = mode;
}
int Context::GetThreadAffinityMode() const {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->affinity_mode_;
}

void Context::SetThreadAffinity(const std::vector<int> &core_list) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->affinity_core_list_ = core_list;
}
std::vector<int32_t> Context::GetThreadAffinityCoreList() const {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->affinity_core_list_;
}

std::vector<std::shared_ptr<DeviceInfoContext>> &Context::MutableDeviceInfo() {
  MS_EXCEPTION_IF_NULL(data_);
  return data_->device_info_list;
}

DeviceInfoContext::DeviceInfoContext() : data_(std::make_shared<Data>()) {}

void CPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionCpuEnableFP16] = is_fp16;
}
bool CPUDeviceInfo::GetEnableFP16() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<bool>(data_, kModelOptionCpuEnableFP16);
}

void GPUDeviceInfo::SetEnableFP16(bool is_fp16) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionGPUEnableFP16] = is_fp16;
}
bool GPUDeviceInfo::GetEnableFP16() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<bool>(data_, kModelOptionGPUEnableFP16);
}

void KirinNPUDeviceInfo::SetFrequency(int frequency) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionKirinNpuFrequency] = frequency;
}
int KirinNPUDeviceInfo::GetFrequency() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<int>(data_, kModelOptionKirinNpuFrequency);
}

void GPUDeviceInfo::SetDeviceID(uint32_t device_id) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionGPUDeviceID] = device_id;
}

uint32_t GPUDeviceInfo::GetDeviceID() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<uint32_t>(data_, kModelOptionGPUDeviceID);
}

int GPUDeviceInfo::GetRankID() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

int GPUDeviceInfo::GetGroupSize() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

void GPUDeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionGPUPrecisionMode] = CharToString(precision_mode);
}
std::vector<char> GPUDeviceInfo::GetPrecisionModeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionGPUPrecisionMode);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetDeviceID(uint32_t device_id) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310DeviceID] = device_id;
}
uint32_t AscendDeviceInfo::GetDeviceID() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<uint32_t>(data_, kModelOptionAscend310DeviceID);
}

void AscendDeviceInfo::SetInsertOpConfigPath(const std::vector<char> &cfg_path) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InsertOpCfgPath] = CharToString(cfg_path);
}
std::vector<char> AscendDeviceInfo::GetInsertOpConfigPathChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InsertOpCfgPath);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetInputFormat(const std::vector<char> &format) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InputFormat] = CharToString(format);
}
std::vector<char> AscendDeviceInfo::GetInputFormatChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InputFormat);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetInputShape(const std::vector<char> &shape) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InputShape] = CharToString(shape);
}
std::vector<char> AscendDeviceInfo::GetInputShapeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310InputShape);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size) {
  MS_EXCEPTION_IF_NULL(data_);
  std::string batchs = "";
  for (size_t i = 0; i < dynamic_batch_size.size(); ++i) {
    if (i != 0) {
      batchs.push_back(',');
    }
    batchs += std::to_string(dynamic_batch_size[i]);
  }
  data_->params[kModelOptionAscend310DynamicBatchSize] = batchs;
}
std::vector<char> AscendDeviceInfo::GetDynamicBatchSizeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310DynamicBatchSize);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetDynamicImageSize(const std::vector<char> &) { return; }

std::vector<char> AscendDeviceInfo::GetDynamicImageSizeChar() const { return std::vector<char>(); }

void AscendDeviceInfo::SetPrecisionMode(const std::vector<char> &precision_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310PrecisionMode] = CharToString(precision_mode);
}
std::vector<char> AscendDeviceInfo::GetPrecisionModeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310PrecisionMode);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetOpSelectImplMode(const std::vector<char> &op_select_impl_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310OpSelectImplMode] = CharToString(op_select_impl_mode);
}
std::vector<char> AscendDeviceInfo::GetOpSelectImplModeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310OpSelectImplMode);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetFusionSwitchConfigPath(const std::vector<char> &cfg_path) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[KModelOptionAscend310FusionSwitchCfgPath] = CharToString(cfg_path);
}
std::vector<char> AscendDeviceInfo::GetFusionSwitchConfigPathChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, KModelOptionAscend310FusionSwitchCfgPath);
  return StringToChar(ref);
}

void AscendDeviceInfo::SetInputShapeMap(const std::map<int, std::vector<int>> &shape) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310InputShapeMap] = shape;
}
std::map<int, std::vector<int>> AscendDeviceInfo::GetInputShapeMap() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<std::map<int, std::vector<int>>>(data_, kModelOptionAscend310InputShapeMap);
}

void AscendDeviceInfo::SetOutputType(enum DataType output_type) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310OutputType] = output_type;
}
enum DataType AscendDeviceInfo::GetOutputType() const {
  MS_EXCEPTION_IF_NULL(data_);
  return GetValue<enum DataType>(data_, kModelOptionAscend310OutputType);
}

void AscendDeviceInfo::SetBufferOptimizeMode(const std::vector<char> &buffer_optimize_mode) {
  MS_EXCEPTION_IF_NULL(data_);
  data_->params[kModelOptionAscend310BufferOptimize] = CharToString(buffer_optimize_mode);
}
std::vector<char> AscendDeviceInfo::GetBufferOptimizeModeChar() const {
  MS_EXCEPTION_IF_NULL(data_);
  const std::string &ref = GetValue<std::string>(data_, kModelOptionAscend310BufferOptimize);
  return StringToChar(ref);
}
}  // namespace mindspore
