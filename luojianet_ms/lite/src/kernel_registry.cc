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
#include "src/kernel_registry.h"
#include <utility>
#include <memory>
#include "include/errorcode.h"
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
#include "include/registry/register_kernel.h"
#endif
#include "src/ops/populate/populate_register.h"
#include "src/common/version_manager.h"
#include "nnacl/pooling_parameter.h"
#if defined(ENABLE_FP16) && defined(ENABLE_ARM)
#if defined(__ANDROID__)
#include <asm/hwcap.h>
#endif
#include "common/utils.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#endif
#include "src/common/tensor_util.h"

using luojianet_ms::kernel::kBuiltin;
using luojianet_ms::kernel::kCPU;
using luojianet_ms::kernel::KERNEL_ARCH;
using luojianet_ms::kernel::KernelCreator;
using luojianet_ms::kernel::KernelKey;
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
using luojianet_ms::registry::CreateKernel;
using luojianet_ms::registry::KernelDesc;
#endif

namespace luojianet_ms::lite {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
namespace {
constexpr auto kArchCPU = "CPU";
constexpr auto kArchGPU = "GPU";
void KernelKeyToKernelDesc(const KernelKey &key, KernelDesc *desc) {
  MS_ASSERT(desc != nullptr);
  desc->data_type = static_cast<DataType>(key.data_type);
  desc->type = key.type;
  desc->arch = key.kernel_arch;
  desc->provider = key.provider;
}
}  // namespace
#endif

void KernelRegistry::CreatorArraysInit() {
  std::unique_lock<std::mutex> malloc_creator_array(lock_);
  if (creator_arrays_ == nullptr) {
    creator_arrays_ = reinterpret_cast<KernelCreator *>(malloc(array_size_ * sizeof(KernelCreator)));
    if (creator_arrays_ != nullptr) {
      memset(creator_arrays_, 0, array_size_ * sizeof(KernelCreator));
    }
  }
  if (inner_op_creator_arrays_ == nullptr) {
    inner_op_creator_arrays_ = reinterpret_cast<KernelCreator *>(malloc(inner_op_array_size_ * sizeof(KernelCreator)));
    if (inner_op_creator_arrays_ != nullptr) {
      memset(inner_op_creator_arrays_, 0, inner_op_array_size_ * sizeof(KernelCreator));
    }
  }
  return;
}

KernelRegistry *KernelRegistry::GetInstance() {
  static KernelRegistry instance;
  return &instance;
}

kernel::KernelCreator KernelRegistry::GetCreator(const KernelKey &desc) {
  if (desc.provider == kBuiltin) {
    int index = GetCreatorFuncIndex(desc);
    if (desc.type >= PrimType_MIN && desc.type < PrimType_MAX) {
      if (index >= array_size_ || index < 0) {
        MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type " << desc.data_type << ",op type "
                      << desc.type;
        return nullptr;
      }
      if (creator_arrays_ != nullptr) {
        return creator_arrays_[index];
      }
    } else if (desc.type >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
      MS_CHECK_TRUE_RET(index >= 0 && index < inner_op_array_size_, nullptr);
      if (inner_op_creator_arrays_ != nullptr) {
        return inner_op_creator_arrays_[index];
      }
    }
  }
  MS_LOG(ERROR) << "Call wrong interface!provider: " << desc.provider;
  return nullptr;
}

int KernelRegistry::GetCreatorFuncIndex(const kernel::KernelKey desc) {
  int device_index = static_cast<int>(desc.arch) - kKernelArch_MIN;
  int dType_index = static_cast<int>(desc.data_type) - kNumberTypeBegin;
  int op_index = static_cast<int>(desc.type);
  int op_type_length = op_type_length_;
  if (op_index >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
    op_type_length = inner_op_type_length_;
    op_index -= PrimType_InnerOpMin;
  }
  int index = device_index * data_type_length_ * op_type_length + dType_index * op_type_length + op_index;
  return index;
}

void KernelRegistry::RegKernel(const KernelKey desc, const kernel::KernelCreator creator) {
  CreatorArraysInit();
  int index = GetCreatorFuncIndex(desc);
  if (desc.type >= PrimType_MIN && desc.type < PrimType_MAX) {
    if (index >= array_size_ || index < 0) {
      MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                    << desc.type;
      return;
    }
    if (creator_arrays_ != nullptr) {
      creator_arrays_[index] = creator;
    }
  } else if (desc.type >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
    MS_CHECK_TRUE_RET_VOID(index >= 0 && index < inner_op_array_size_);
    if (inner_op_creator_arrays_ != nullptr) {
      inner_op_creator_arrays_[index] = creator;
    }
  }
}

void KernelRegistry::RegKernel(KERNEL_ARCH arch, TypeId data_type, int op_type, kernel::KernelCreator creator) {
  CreatorArraysInit();
  KernelKey desc = {arch, data_type, op_type};
  int index = GetCreatorFuncIndex(desc);
  if (desc.type >= PrimType_MIN && desc.type < PrimType_MAX) {
    if (index >= array_size_ || index < 0) {
      MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                    << desc.type;
      return;
    }
    if (creator_arrays_ != nullptr) {
      creator_arrays_[index] = creator;
    }
  } else if (desc.type >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
    MS_CHECK_TRUE_RET_VOID(index >= 0 && index < inner_op_array_size_);
    if (inner_op_creator_arrays_ != nullptr) {
      inner_op_creator_arrays_[index] = creator;
    }
  }
}

KernelRegistry::~KernelRegistry() {
  KernelRegistry *instance = GetInstance();
  std::unique_lock<std::mutex> malloc_creator_array(instance->lock_);
  if (instance->creator_arrays_ != nullptr) {
    free(instance->creator_arrays_);
    instance->creator_arrays_ = nullptr;
  }
  if (instance->inner_op_creator_arrays_ != nullptr) {
    free(instance->inner_op_creator_arrays_);
    instance->inner_op_creator_arrays_ = nullptr;
  }
}

bool KernelRegistry::SupportKernel(const KernelKey &key) {
  auto kernel_creator = GetCreator(key);
  return kernel_creator != nullptr;
}

#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
int KernelRegistry::GetCustomKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                    const luojianet_ms::Context *ms_ctx, const kernel::KernelKey &key,
                                    kernel::LiteKernel **kernel, const void *primitive) {
  MS_ASSERT(ms_ctx != nullptr);
  MS_ASSERT(kernel != nullptr);
  KernelDesc desc;
  KernelKeyToKernelDesc(key, &desc);
  auto creator = registry::RegisterKernel::GetCreator(static_cast<const schema::Primitive *>(primitive), &desc);
  if (creator == nullptr) {
    return RET_NOT_SUPPORT;
  }

  auto base_kernel = creator(LiteTensorsToMSTensors(in_tensors), LiteTensorsToMSTensors(out_tensors),
                             static_cast<const schema::Primitive *>(primitive), ms_ctx);
  if (base_kernel != nullptr) {
    auto *lite_kernel = new (std::nothrow) kernel::LiteKernel(base_kernel);
    if (lite_kernel != nullptr) {
      kernel::KernelKey tmp_key = key;
      if (desc.arch == kArchCPU) {
        tmp_key.arch = kernel::kCPU;
      } else if (desc.arch == kArchGPU) {
        tmp_key.arch = kernel::kGPU;
      } else {
        tmp_key.arch = kernel::kCustom;
      }
      lite_kernel->set_desc(tmp_key);
      *kernel = lite_kernel;
      return RET_OK;
    }
  }
  return RET_ERROR;
}
#endif

int KernelRegistry::GetKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                              const InnerContext *ctx, const luojianet_ms::Context *ms_ctx, const kernel::KernelKey &key,
                              OpParameter *parameter, kernel::LiteKernel **kernel, const void *primitive) {
  MS_ASSERT(ctx != nullptr);
  MS_ASSERT(kernel != nullptr);
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  if (key.provider == kBuiltin) {
#endif
    auto creator = GetCreator(key);
    if (creator != nullptr) {
      auto inner_kernel = creator(in_tensors, out_tensors, parameter, ctx, key);
      if (inner_kernel != nullptr) {
        inner_kernel->set_registry_data_type(key.data_type);
        std::shared_ptr<kernel::Kernel> shared_kernel(inner_kernel);
        auto *lite_kernel = new (std::nothrow) kernel::LiteKernel(shared_kernel);
        if (lite_kernel != nullptr) {
          lite_kernel->set_desc(key);
          lite_kernel->set_context(ctx);
          *kernel = lite_kernel;
          return RET_OK;
        }
      }
      return RET_ERROR;
    }
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  } else {
    auto ret = GetCustomKernel(in_tensors, out_tensors, ms_ctx, key, kernel, primitive);
    if (ret == RET_OK) {
      (*kernel)->set_context(ctx);
    }
    return ret;
  }
#endif
  return RET_NOT_SUPPORT;
}
}  // namespace luojianet_ms::lite
