/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "include/api/graph.h"
#include "include/api/serialization.h"
#include "include/api/types.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_luojianet_Graph_loadModel(JNIEnv *env, jobject thiz, jstring ms_file) {
  auto graph = new (std::nothrow) luojianet_ms::Graph();
  if (graph == nullptr) {
    MS_LOGE("Model new failed");
    return jlong(nullptr);
  }
  auto status =
    luojianet_ms::Serialization::Load(env->GetStringUTFChars(ms_file, JNI_FALSE), luojianet_ms::ModelType::kMindIR, graph);
  if (status != luojianet_ms::kSuccess) {
    MS_LOGE("Load graph from file failed");
    delete graph;
    return jlong(nullptr);
  }
  return jlong(graph);
}

extern "C" JNIEXPORT void JNICALL Java_com_luojianet_Graph_free(JNIEnv *env, jobject thiz, jlong graph_ptr) {
  auto *pointer = reinterpret_cast<void *>(graph_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return;
  }
  auto *lite_graph_ptr = static_cast<luojianet_ms::Graph *>(pointer);
  delete (lite_graph_ptr);
}
