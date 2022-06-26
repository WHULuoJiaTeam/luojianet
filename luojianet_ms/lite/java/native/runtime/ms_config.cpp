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

#include <jni.h>
#include "common/ms_log.h"
#include "include/context.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_luojianet_ms_lite_config_MSConfig_createMSConfig(
  JNIEnv *env, jobject thiz, jint device_type, jint thread_num, jint cpu_bind_mode, jboolean enable_float16) {
  auto *context = new (std::nothrow) luojianet_ms::lite::Context();
  if (context == nullptr) {
    MS_LOGE("new Context fail!");
    return (jlong) nullptr;
  }

  auto &cpu_device_ctx = context->device_list_[0];
  switch (cpu_bind_mode) {
    case 0:
      cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = luojianet_ms::lite::NO_BIND;
      break;
    case 1:
      cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = luojianet_ms::lite::HIGHER_CPU;
      break;
    case 2:
      cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = luojianet_ms::lite::MID_CPU;
      break;
    default:
      MS_LOGE("Invalid cpu_bind_mode : %d", cpu_bind_mode);
      delete context;
      return (jlong) nullptr;
  }
  if (enable_float16) {
    cpu_device_ctx.device_info_.cpu_device_info_.enable_float16_ = true;
  }
  switch (device_type) {
    case 0:
      context->device_list_[0].device_type_ = luojianet_ms::lite::DT_CPU;
      break;
    case 1:  // DT_GPU
    {
      luojianet_ms::lite::DeviceContext gpu_device_ctx{luojianet_ms::lite::DT_GPU, {false}};
      if (enable_float16) {
        gpu_device_ctx.device_info_.gpu_device_info_.enable_float16_ = true;
      }
      context->device_list_.push_back(gpu_device_ctx);
      break;
    }
    case 2:  // DT_NPU
      MS_LOGE("We only support CPU and GPU now.");
      delete context;
      return (jlong) nullptr;
    default:
      MS_LOGE("Invalid device_type : %d", device_type);
      delete context;
      return (jlong) nullptr;
  }
  context->thread_num_ = thread_num;
  return (jlong)context;
}

extern "C" JNIEXPORT void JNICALL Java_com_luojianet_ms_lite_config_MSConfig_free(JNIEnv *env, jobject thiz,
                                                                               jlong context_ptr) {
  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return;
  }
  auto *lite_context_ptr = static_cast<luojianet_ms::lite::Context *>(pointer);
  delete (lite_context_ptr);
}
