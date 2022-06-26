/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <set>
#include <algorithm>
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/memory_manager.h"
#include "runtime/device/convert_tensor_utils.h"
#include "plugin/device/ascend/hal/device/ascend_launch_transdata.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_context.h"
#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "abstract/utils.h"
#include "include/common/utils/utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#endif

namespace {
const std::unordered_map<mindspore::TypeId, std::string> type_id_name_map = {
  {mindspore::kNumberTypeBool, "bool"},       {mindspore::kNumberTypeInt8, "int8"},
  {mindspore::kNumberTypeInt16, "int16"},     {mindspore::kNumberTypeInt32, "int32"},
  {mindspore::kNumberTypeInt64, "int64"},     {mindspore::kNumberTypeFloat16, "float16"},
  {mindspore::kNumberTypeFloat32, "float32"}, {mindspore::kNumberTypeUInt8, "uint8"},
  {mindspore::kNumberTypeUInt16, "uint16"},   {mindspore::kNumberTypeUInt32, "uint32"},
  {mindspore::kNumberTypeUInt64, "uint64"}};
const std::set<std::pair<std::string, std::string>> use_trans_data = {
  std::make_pair("float16", mindspore::kOpFormat_NC1HWC0), std::make_pair("float32", mindspore::kOpFormat_NC1HWC0),
  std::make_pair("bool", mindspore::kOpFormat_NC1HWC0),    std::make_pair("float32", mindspore::kOpFormat_FRAC_Z),
  std::make_pair("float16", mindspore::kOpFormat_FRAC_Z),  std::make_pair("float16", mindspore::kOpFormat_FRAC_NZ),
  std::make_pair("float32", mindspore::kOpFormat_FRAC_NZ), std::make_pair("int32", mindspore::kOpFormat_FRAC_NZ),
  std::make_pair("float16", mindspore::kOpFormat_NHWC),    std::make_pair("float32", mindspore::kOpFormat_NHWC),
  std::make_pair("int8", mindspore::kOpFormat_NHWC),       std::make_pair("int16", mindspore::kOpFormat_NHWC),
  std::make_pair("int32", mindspore::kOpFormat_NHWC),      std::make_pair("int64", mindspore::kOpFormat_NHWC),
  std::make_pair("uint8", mindspore::kOpFormat_NHWC),      std::make_pair("uint16", mindspore::kOpFormat_NHWC),
  std::make_pair("uint32", mindspore::kOpFormat_NHWC),     std::make_pair("uint64", mindspore::kOpFormat_NHWC),
  std::make_pair("float16", mindspore::kOpFormat_HWCN),    std::make_pair("float32", mindspore::kOpFormat_HWCN),
  std::make_pair("int8", mindspore::kOpFormat_HWCN),       std::make_pair("int16", mindspore::kOpFormat_HWCN),
  std::make_pair("int32", mindspore::kOpFormat_HWCN),      std::make_pair("int64", mindspore::kOpFormat_HWCN),
  std::make_pair("uint8", mindspore::kOpFormat_HWCN),      std::make_pair("uint16", mindspore::kOpFormat_HWCN),
  std::make_pair("uint32", mindspore::kOpFormat_HWCN),     std::make_pair("uint64", mindspore::kOpFormat_HWCN)};
}  // namespace

namespace mindspore {
namespace device {
namespace ascend {
const int FLOAT_LEN = sizeof(float);
const int FLOAT16_LEN = 2;
const std::set<std::string> kOpNeedTransFormat = {
  kOpFormat_NHWC,    kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z,   kOpFormat_C1HWNCoC0,
  kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};

void SyncMemory(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind) {
  if (size == 0) {
    return;
  }
  if (dst == nullptr) {
    MS_LOG(EXCEPTION) << "dst ptr is null, please check the address is set correctly.";
  }
  if (src == nullptr) {
    MS_LOG(EXCEPTION) << "src ptr is null, please check the address is set correctly.";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->SetContext();

  // Only apply asynchronous copy in Pynative && ACL_MEMCPY_HOST_TO_DEVICE mode
  if (execution_mode != kPynativeMode || kind != ACL_MEMCPY_HOST_TO_DEVICE) {
    auto ret_rt_memcpy = aclrtMemcpy(dst, size, src, size, kind);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "aclrtMemcpy failed";
    }
  } else {
    auto ret = runtime_instance->MemcpyAsync(dst, src, size, static_cast<int32_t>(RT_MEMCPY_HOST_TO_DEVICE_EX));
    if (!ret) {
      MS_EXCEPTION(DeviceProcessError) << "MemcpyAsync failed";
    }
  }
}

bool DataSync(void *dst, const void *src, uint64_t src_size) {
  if (dst == src) {
    MS_LOG(INFO) << "dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->MemcpyAsync(dst, src, src_size, static_cast<int32_t>(RT_MEMCPY_DEVICE_TO_DEVICE));
  if (!ret) {
    MS_LOG(ERROR) << "rtMemcpyAsync failed!";
    return false;
  }
  return true;
}

bool FloatToHalfAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / FLOAT_LEN;
  if (elem_num != (dst_size / FLOAT16_LEN)) {
    MS_EXCEPTION(ArgumentError) << "FloatToHalf failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  FloatToHalf(half_data.data(), src, elem_num);
  SyncMemory(dst, half_data.data(), dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool Float64ToFloatAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size / 2 != dst_size) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = dst_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  DoubleToFloat(host_tmp.data(), src, elem_num);
  SyncMemory(dst, host_tmp.data(), dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool SyncDeviceToHostAndHalfToFloat(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / FLOAT16_LEN;
  if (elem_num != (dst_size / FLOAT_LEN)) {
    MS_EXCEPTION(ArgumentError) << "HalfToFloat failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  SyncMemory(half_data.data(), src, src_size, ACL_MEMCPY_DEVICE_TO_HOST);
  HalfToFloat(dst, half_data.data(), elem_num);
  return true;
}

bool SyncDeviceToHostAndFloatToFloat64(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size != dst_size / 2) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = src_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  SyncMemory(host_tmp.data(), src, src_size, ACL_MEMCPY_DEVICE_TO_HOST);
  FloatToDouble(dst, host_tmp.data(), elem_num);
  return true;
}

void AscendDeviceAddress::BindDevice() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    return;
  }

  // Bind device by device name and device id on the current thread.
  if (device_name_ != "") {
    auto device_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
    auto ascend_device_context = dynamic_cast<AscendDeviceContext *>(device_context);
    MS_EXCEPTION_IF_NULL(ascend_device_context);
    ascend_device_context->BindDeviceToCurrentThread();
  } else {
    MS_LOG(WARNING) << "device name is null.";
  }
}

void AscendDeviceAddress::SyncStream() const {
  MS_LOG(DEBUG) << "SyncStream Start!";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  MS_LOG(DEBUG) << "SyncStream Finish!";
}

bool AscendDeviceAddress::SyncDeviceToHost(size_t size, void *const host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  BindDevice();
  SyncStream();
  SyncMemory(host_ptr, ptr_, size, ACL_MEMCPY_DEVICE_TO_HOST);
  return true;
}

bool AscendDeviceAddress::SyncHostToDevice(size_t size, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  BindDevice();
  SyncMemory(ptr_, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHost(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           void *host_ptr) const {
  MS_LOG(INFO) << "SyncDeviceToHost, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  SyncStream();
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW) {
    if (type_id_ == type) {
      SyncMemory(host_ptr, ptr_, size, ACL_MEMCPY_DEVICE_TO_HOST);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = SyncDeviceToHostAndFloatToFloat64(host_ptr, size, ptr_, size_);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      auto host = std::vector<uint8_t>(size_);
      SyncMemory(host.data(), ptr_, size_, ACL_MEMCPY_DEVICE_TO_HOST);
      const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size_};
      sync_ok = trans::TransDataType(type_args, host_ptr);
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
    }
  } else {
    auto iter = kOpNeedTransFormat.find(format_);
    if (iter != kOpNeedTransFormat.end()) {
      sync_ok = SyncDeviceToHostAndConvertFormat(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported to trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

ShapeVector AscendDeviceAddress::GetDeviceShape(ShapeVector *host_shape) const {
  MS_EXCEPTION_IF_NULL(host_shape);
  ShapeVector device_shape;
  auto node_index = GetNodeIndex();
  if (format_ == kOpFormat_FRAC_NZ || format_ == kOpFormat_NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, format_, node_index.first, node_index.second, type_id_);
  } else {
    if (!host_shape_.empty()) {
      host_shape->clear();
      *host_shape = host_shape_;
    }
    *host_shape = trans::PaddingShape(*host_shape, format_);
    device_shape = trans::TransShapeToDevice(*host_shape, format_, node_index.first, node_index.second, type_id_);
  }
  return device_shape;
}

std::shared_ptr<LaunchKernel> AscendDeviceAddress::CreateLaunchTransData(const std::vector<size_t> &host_shape,
                                                                         const std::string &ori_format,
                                                                         const std::string &dst_format) const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetCurrentKernelRuntime();
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto stream = runtime_instance->compute_stream();
  auto node = GetNodeIndex();
  int64_t groups = 1;
  if (format_ == kOpFormat_FRAC_Z && node.first != nullptr) {
    groups = common::AnfAlgo::GetAttrGroups(node.first, node.second);
  }
  auto launch_trans_data =
    std::make_shared<AscendLaunchTransData>(stream, type_id_, size_, ori_format, dst_format, host_shape, groups);
  MS_EXCEPTION_IF_NULL(launch_trans_data);
  return launch_trans_data;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormatBasedOnTransData(const std::vector<size_t> &host_shape,
                                                                           size_t size, mindspore::TypeId type,
                                                                           void *host_ptr) const {
  bool sync_ok = true;
  const std::string dst_format = kOpFormat_NCHW;
  if (launch_transdata_ == nullptr) {
    launch_transdata_ = CreateLaunchTransData(host_shape, format_, dst_format);
    MS_EXCEPTION_IF_NULL(launch_transdata_);
  }
  // launch transdata
  launch_transdata_->SetInputAddr(static_cast<uint8_t *>(ptr_));
  launch_transdata_->LaunchOpKernel();
  SyncStream();
  auto output_addr_vec = launch_transdata_->GetKernelOutputAddr();
  if (output_addr_vec.size() != 1) {
    launch_transdata_->FreeLaunchDeviceMem();
    MS_LOG(EXCEPTION) << "Launch transdata outputs should have only one output";
    return false;
  }
  if (type_id_ == type) {
    SyncMemory(host_ptr, output_addr_vec[0], size, ACL_MEMCPY_DEVICE_TO_HOST);
  } else {
    auto host = std::vector<uint8_t>(size);
    SyncMemory(host.data(), output_addr_vec[0], size, ACL_MEMCPY_DEVICE_TO_HOST);
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), SizeToLong(shape_size), type_id_, type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      launch_transdata_->FreeLaunchDeviceMem();
      return false;
    }
  }
  launch_transdata_->FreeLaunchDeviceMem();
  return sync_ok;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormat(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, void *host_ptr) const {
  MS_LOG(INFO) << "SyncDeviceToHostAndConvertFormat, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  auto device_shape = GetDeviceShape(&host_shape);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode &&
      type_id_name_map.find(type_id_) != type_id_name_map.end()) {
    std::pair<std::string, std::string> type_format = std::make_pair(type_id_name_map.at(type_id_), format_);
    if (use_trans_data.find(type_format) != use_trans_data.end()) {
      std::vector<size_t> st_shape;
      (void)std::transform(host_shape.begin(), host_shape.end(), std::back_inserter(st_shape), LongToSize);
      sync_ok = SyncDeviceToHostAndConvertFormatBasedOnTransData(st_shape, size, type, host_ptr);
      return sync_ok;
    }
  }
  auto host_tmp = std::vector<uint8_t>(size_);
  SyncMemory(host_tmp.data(), ptr_, size_, ACL_MEMCPY_DEVICE_TO_HOST);
  auto node_index = GetNodeIndex();
  if (type_id_ != type) {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto host = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
  } else {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host_ptr, node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDevice(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           const void *host_ptr, const std::string &format) const {
  MS_LOG(INFO) << "SyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(format:" << format << ", type_id:" << TypeIdLabel(type)
               << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW || format_ == format) {
    if (type_id_ == type) {
      SyncMemory(ptr_, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = Float64ToFloatAndSyncHostToDevice(ptr_, size_, host_ptr, size);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
      auto host_tmp = std::vector<uint8_t>(size_);
      sync_ok = trans::TransDataType(type_args, host_tmp.data());
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
      SyncMemory(ptr_, host_tmp.data(), size_, ACL_MEMCPY_HOST_TO_DEVICE);
    }
  } else {
    auto iter = kOpNeedTransFormat.find(format_);
    if (iter != kOpNeedTransFormat.end()) {
      sync_ok = ConvertFormatAndSyncHostToDevice(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncDeviceToDeviceWithSameFormatType(const ShapeVector &shape, size_t size, TypeId type,
                                                               const void *src_ptr, const std::string &format) const {
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }

  if (size_ < size) {
    MS_LOG(ERROR) << "src size is greater than det size, src size is: " << size << ", dst size is: " << size_;
    return false;
  }
  if (format_ != format || type_id_ != type) {
    MS_LOG(ERROR) << "format or type is different, src(format:" << format << ", type_id:" << TypeIdLabel(type)
                  << "), dst(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_);
    return false;
  }
  BindDevice();
  return AsyncDeviceToDevice(shape, size, type, src_ptr, format);
}

bool AscendDeviceAddress::SyncDeviceToDeviceWithDiffFormatType(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }

  auto src_device_address = dynamic_cast<const AscendDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_device_address);
  BindDevice();
  auto host_shape = src_device_address->host_shape();
  if (host_shape.empty()) {
    MS_LOG(ERROR) << "host shape is empty, please check whether the host shape of source device address"
                  << src_device_address << " is set.";
    return false;
  }
  auto host_tensor = std::make_shared<tensor::Tensor>(src_device_address->type_id(), host_shape);
  auto host_tensor_size = LongToSize(host_tensor->data().nbytes());
  auto host_tensor_type = host_tensor->data_type();
  if (!src_device_address->SyncDeviceToHost(host_shape, host_tensor_size, host_tensor_type, host_tensor->data_c())) {
    MS_LOG(ERROR) << "Sync device to device failed at the stage of sync device to intermediate Tensor.";
    return false;
  }
  if (!SyncHostToDevice(host_shape, host_tensor_size, host_tensor_type, host_tensor->data_c(),
                        host_tensor->device_info().host_format_)) {
    MS_LOG(ERROR) << "Sync device to device failed at the stage of sync intermediate tensor to device.";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::SyncDeviceToDevice(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  auto src_device_address = dynamic_cast<const AscendDeviceAddress *>(src_device_addr);
  if (format_ == src_device_address->format() && type_id_ == src_device_address->type_id()) {
    return SyncDeviceToDeviceWithSameFormatType(ShapeVector(), src_device_address->GetSize(),
                                                src_device_address->type_id(), src_device_address->GetPtr(),
                                                src_device_address->format());
  } else {
    MS_LOG(WARNING) << "Can not copy from device to device directly, format or type is different, src(format:"
                    << src_device_address->format() << ", type_id:" << TypeIdLabel(src_device_address->type_id())
                    << "), dst(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
                    << ", use the intermediate Tensor copy instead.";
    return SyncDeviceToDeviceWithDiffFormatType(src_device_addr);
  }
}

bool AscendDeviceAddress::AsyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                                              const std::string &format) const {
  MS_LOG(INFO) << "AsyncDeviceToDevice, dst(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), src(format:" << format << ", type_id:" << TypeIdLabel(type)
               << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  bool sync_ok = false;
  if (format_ == format && type_id_ == type) {
    if (!DataSync(ptr_, src_ptr, size)) {
      MS_LOG(ERROR) << "DataSync failed!";
      return false;
    }
    sync_ok = true;
  } else {
    MS_LOG(ERROR) << "format or type is different.";
    sync_ok = false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::ConvertFormatAndSyncHostToDevice(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, const void *host_ptr) const {
  bool sync_ok = false;
  MS_LOG(INFO) << "ConvertFormatAndSyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  auto node_index = GetNodeIndex();
  std::vector<int64_t> device_shape;
  if (format_ == kOpFormat_FRAC_NZ) {
    device_shape = trans::TransShapeToDevice(host_shape, format_, node_index.first, node_index.second, type_id_);
  } else {
    host_shape = trans::PaddingShape(host_shape, format_);
    device_shape = trans::TransShapeToDevice(host_shape, format_, node_index.first, node_index.second, type_id_);
  }
  if (type_id_ != type) {
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransDataType(type_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto dst_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, dst_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    SyncMemory(ptr_, dst_tmp.data(), size_, ACL_MEMCPY_HOST_TO_DEVICE);
  } else {
    const trans::FormatArgs format_args{host_ptr, size_, kOpFormat_NCHW, format_, host_shape, device_shape, type_id_};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, host_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    SyncMemory(ptr_, host_tmp.data(), size_, ACL_MEMCPY_HOST_TO_DEVICE);
  }
  return sync_ok;
}

void AscendDeviceAddress::ClearDeviceMemory() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    if (communication_ptr_ != nullptr) {
      AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr_);
      communication_ptr_ = nullptr;
    } else {
      AscendMemoryPool::GetInstance().FreeTensorMem(ptr_);
    }
    ptr_ = nullptr;
  }
}

AscendDeviceAddress::~AscendDeviceAddress() {
  try {
    ClearDeviceMemory();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "AscendDeviceAddress destructor failed: " << e.what();
  }
}

#ifndef ENABLE_SECURITY
/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump tensor data to file for e2e dump.
 */
bool AscendDeviceAddress::DumpMemToFile(const std::string &filepath, const std::string &host_fmt,
                                        const ShapeVector &host_shape, TypeId host_type, bool trans_flag) const {
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  if (trans_flag) {
    std::string path = filepath + '.' + host_fmt;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin ||
        host_type == kNumberTypeComplex64) {
      MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
      return false;
    }
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
    MS_EXCEPTION_IF_NULL(out_tensor);
    size_t host_size = out_tensor->data().nbytes();
    ret = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
    ret = DumpJsonParser::DumpToFile(path, out_tensor->data_c(), host_size, host_shape, host_type);
  } else {
    auto host_tmp = std::vector<uint8_t>(size_);
    BindDevice();
    SyncStream();
    auto ret_rt_memcpy = aclrtMemcpy(host_tmp.data(), size_, ptr_, size_, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: aclrtMemcpy mem size[" << size_ << "] fail, ret[" << ret_rt_memcpy << "]";
      return false;
    }
    std::string path = filepath + '.' + format_;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), size_, host_shape, type_id_);
  }

  return ret;
}
#endif

#ifdef ENABLE_DEBUGGER
/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Load tensor to host and create tensor_data object for the loaded tensor.
 */
bool AscendDeviceAddress::LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &,
                                        const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev,
                                        uint32_t root_graph_id, bool force_update) const {
  bool ret = false;
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->TensorExistsInCurrent(tensor_name) && !force_update) {
    MS_LOG(INFO) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }
  // TensorData is freed up in AscendSession class
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  MS_EXCEPTION_IF_NULL(tensor_data);
  tensor_data->SetName(tensor_name);
  tensor_data->SetExecutionOrder(execution_order);
  tensor_data->SetSlot(slot);

  if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin || host_type == kNumberTypeComplex64) {
    MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
    return false;
  }
  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
  MS_EXCEPTION_IF_NULL(out_tensor);
  size_t host_size = out_tensor->data().nbytes();
  auto ret_sync = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
  if (!ret_sync) {
    MS_LOG(ERROR) << "Copy device mem to host failed";
    return ret;
  }
  MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  tensor_data->SetByteSize(LongToSize(out_tensor->data().nbytes()));
  tensor_data->SetType((unsigned int)host_type);
  tensor_data->SetShape(out_tensor->shape());
  tensor_data->SetRootGraphId(root_graph_id);
  ret = debugger->LoadNewTensor(tensor_data, keep_prev);
  return ret;
}
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
