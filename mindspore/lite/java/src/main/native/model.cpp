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

#include "include/api/model.h"
#include <jni.h>
#include "common/ms_log.h"
#include "include/api/serialization.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_buildByGraph(JNIEnv *env, jobject thiz, jlong graph_ptr,
                                                                         jlong context_ptr, jlong cfg_ptr) {
  auto *c_graph_ptr = reinterpret_cast<mindspore::Graph *>(graph_ptr);
  if (c_graph_ptr == nullptr) {
    MS_LOGE("Graph pointer from java is nullptr");
    return jlong(nullptr);
  }

  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOGE("Make context failed");
    return jlong(nullptr);
  }
  context.reset(c_context_ptr);

  auto *c_cfg_ptr = reinterpret_cast<mindspore::TrainCfg *>(cfg_ptr);
  auto cfg = std::make_shared<mindspore::TrainCfg>();
  if (cfg == nullptr) {
    MS_LOGE("Make train config failed");
    return jlong(nullptr);
  }
  if (c_cfg_ptr != nullptr) {
    cfg.reset(c_cfg_ptr);
  } else {
    cfg.reset();
  }
  auto model = new (std::nothrow) mindspore::Model();
  if (model == nullptr) {
    MS_LOGE("Model new failed");
    return jlong(nullptr);
  }

  auto status = model->Build(mindspore::GraphCell(*c_graph_ptr), context, cfg);
  if (status != mindspore::kSuccess) {
    MS_LOGE("Error (%d) during build of model", static_cast<int>(status));
    delete model;
    return jlong(nullptr);
  }
  return jlong(model);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_buildByBuffer(JNIEnv *env, jobject thiz,
                                                                          jobject model_buffer, jint model_type,
                                                                          jlong context_ptr, jcharArray key_str,
                                                                          jstring dec_mod, jstring cropto_lib_path) {
  if (model_buffer == nullptr) {
    MS_LOGE("Buffer from java is nullptr");
    return reinterpret_cast<jlong>(nullptr);
  }
  mindspore::ModelType c_model_type;
  if (model_type >= static_cast<int>(mindspore::kMindIR) && model_type <= static_cast<int>(mindspore::kMindIR_Lite)) {
    c_model_type = static_cast<mindspore::ModelType>(model_type);
  } else {
    MS_LOGE("Invalid model type : %d", model_type);
    return (jlong) nullptr;
  }
  jlong buffer_len = env->GetDirectBufferCapacity(model_buffer);
  auto *model_buf = static_cast<char *>(env->GetDirectBufferAddress(model_buffer));

  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOGE("Make context failed");
    return jlong(nullptr);
  }
  context.reset(c_context_ptr);

  auto model = new (std::nothrow) mindspore::Model();
  if (model == nullptr) {
    MS_LOGE("Model new failed");
    return jlong(nullptr);
  }
  auto c_dec_mod = env->GetStringUTFChars(dec_mod, JNI_FALSE);
  mindspore::Status status;
  if (key_str != NULL) {
    jchar *key_array = env->GetCharArrayElements(key_str, NULL);
    auto key_len = static_cast<size_t>(env->GetArrayLength(key_str));
    char *dec_key_data = new (std::nothrow) char[key_len];
    if (dec_key_data == nullptr) {
      MS_LOGE("Dec key new failed");
      delete model;
      return jlong(nullptr);
    }
    for (size_t i = 0; i < key_len; i++) {
      dec_key_data[i] = key_array[i];
    }
    env->ReleaseCharArrayElements(key_str, key_array, JNI_ABORT);
    mindspore::Key dec_key{dec_key_data, key_len};
    auto c_cropto_lib_path = env->GetStringUTFChars(cropto_lib_path, JNI_FALSE);
    status = model->Build(model_buf, buffer_len, c_model_type, context, dec_key, c_dec_mod, c_cropto_lib_path);
  } else {
    status = model->Build(model_buf, buffer_len, c_model_type, context);
  }
  if (status != mindspore::kSuccess) {
    MS_LOGE("Error (%d) during build of model", static_cast<int>(status));
    delete model;
    return jlong(nullptr);
  }
  return jlong(model);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_buildByPath(JNIEnv *env, jobject thiz, jstring model_path,
                                                                        jint model_type, jlong context_ptr,
                                                                        jcharArray key_str, jstring dec_mod,
                                                                        jstring cropto_lib_path) {
  auto c_model_path = env->GetStringUTFChars(model_path, JNI_FALSE);
  mindspore::ModelType c_model_type;
  if (model_type >= static_cast<int>(mindspore::kMindIR) && model_type <= static_cast<int>(mindspore::kMindIR_Lite)) {
    c_model_type = static_cast<mindspore::ModelType>(model_type);
  } else {
    MS_LOGE("Invalid model type : %d", model_type);
    return (jlong) nullptr;
  }
  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOGE("Make context failed");
    return jlong(nullptr);
  }
  context.reset(c_context_ptr);

  auto model = new (std::nothrow) mindspore::Model();
  if (model == nullptr) {
    MS_LOGE("Model new failed");
    return jlong(nullptr);
  }
  auto c_dec_mod = env->GetStringUTFChars(dec_mod, JNI_FALSE);
  mindspore::Status status;
  if (key_str != NULL) {
    jchar *key_array = env->GetCharArrayElements(key_str, NULL);
    auto key_len = static_cast<size_t>(env->GetArrayLength(key_str));
    char *dec_key_data = new (std::nothrow) char[key_len];
    if (dec_key_data == nullptr) {
      MS_LOGE("Dec key new failed");
      delete model;
      return jlong(nullptr);
    }
    for (size_t i = 0; i < key_len; i++) {
      dec_key_data[i] = key_array[i];
    }
    env->ReleaseCharArrayElements(key_str, key_array, JNI_ABORT);
    mindspore::Key dec_key{dec_key_data, key_len};
    auto c_cropto_lib_path = env->GetStringUTFChars(cropto_lib_path, JNI_FALSE);
    status = model->Build(c_model_path, c_model_type, context, dec_key, c_dec_mod, c_cropto_lib_path);
  } else {
    status = model->Build(c_model_path, c_model_type, context);
  }
  if (status != mindspore::kSuccess) {
    MS_LOGE("Error (%d) during build of model", static_cast<int>(status));
    delete model;
    return jlong(nullptr);
  }
  return jlong(model);
}

jobject GetInOrOutTensors(JNIEnv *env, jobject thiz, jlong model_ptr, bool is_input) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<mindspore::Model *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return ret;
  }
  std::vector<mindspore::MSTensor> tensors;
  if (is_input) {
    tensors = pointer->GetInputs();
  } else {
    tensors = pointer->GetOutputs();
  }
  for (auto &tensor : tensors) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
    if (tensor_ptr == nullptr) {
      MS_LOGE("Make ms tensor failed");
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

jlong GetTensorByInOutName(JNIEnv *env, jlong model_ptr, jstring tensor_name, bool is_input) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  mindspore::MSTensor tensor;
  if (is_input) {
    tensor = lite_model_ptr->GetInputByTensorName(env->GetStringUTFChars(tensor_name, JNI_FALSE));
  } else {
    tensor = lite_model_ptr->GetOutputByTensorName(env->GetStringUTFChars(tensor_name, JNI_FALSE));
  }
  if (tensor.impl() == nullptr) {
    return jlong(nullptr);
  }
  auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
  if (tensor_ptr == nullptr) {
    MS_LOGE("Make ms tensor failed");
    return jlong(nullptr);
  }
  return jlong(tensor_ptr.release());
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getInputs(JNIEnv *env, jobject thiz, jlong model_ptr) {
  return GetInOrOutTensors(env, thiz, model_ptr, true);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_getInputByTensorName(JNIEnv *env, jobject thiz,
                                                                                 jlong model_ptr, jstring tensor_name) {
  return GetTensorByInOutName(env, model_ptr, tensor_name, true);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getOutputs(JNIEnv *env, jobject thiz, jlong model_ptr) {
  return GetInOrOutTensors(env, thiz, model_ptr, false);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_getOutputByTensorName(JNIEnv *env, jobject thiz,
                                                                                  jlong model_ptr,
                                                                                  jstring tensor_name) {
  return GetTensorByInOutName(env, model_ptr, tensor_name, false);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getOutputTensorNames(JNIEnv *env, jobject thiz,
                                                                                   jlong model_ptr) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return ret;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto output_names = lite_model_ptr->GetOutputTensorNames();
  for (const auto &output_name : output_names) {
    env->CallBooleanMethod(ret, array_list_add, env->NewStringUTF(output_name.c_str()));
  }
  return ret;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getOutputsByNodeName(JNIEnv *env, jobject thiz,
                                                                                   jlong model_ptr, jstring node_name) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return ret;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto tensors = lite_model_ptr->GetOutputsByNodeName(env->GetStringUTFChars(node_name, JNI_FALSE));
  for (auto &tensor : tensors) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
    if (tensor_ptr == nullptr) {
      MS_LOGE("Make ms tensor failed");
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_getTrainMode(JNIEnv *env, jobject thiz,
                                                                            jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  return static_cast<jboolean>(lite_model_ptr->GetTrainMode());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_setTrainMode(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                            jboolean train_mode) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return jlong(false);
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto status = lite_model_ptr->SetTrainMode(train_mode);
  return static_cast<jboolean>(status.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_runStep(JNIEnv *env, jobject thiz, jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto status = lite_model_ptr->RunStep(nullptr, nullptr);
  return static_cast<jboolean>(status.IsOk());
}

std::vector<mindspore::MSTensor> convertArrayToVector(JNIEnv *env, jlongArray inputs) {
  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  std::vector<mindspore::MSTensor> c_inputs;
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOGE("Tensor pointer from java is nullptr");
      return c_inputs;
    }
    auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_inputs.push_back(*ms_tensor_ptr);
  }
  return c_inputs;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_predict(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                       jlongArray inputs, jlongArray outputs) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto c_inputs = convertArrayToVector(env, inputs);
  auto c_outputs = convertArrayToVector(env, outputs);
  auto status = lite_model_ptr->Predict(c_inputs, &c_outputs);
  return static_cast<jboolean>(status.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_resize(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                      jlongArray inputs, jobjectArray dims) {
  std::vector<std::vector<int64_t>> c_dims;
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);

  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  std::vector<mindspore::MSTensor> c_inputs;
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOGE("Tensor pointer from java is nullptr");
      return (jboolean) false;
    }
    auto &ms_tensor = *static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_inputs.push_back(ms_tensor);
  }
  auto tensor_size = static_cast<int>(env->GetArrayLength(dims));
  for (int i = 0; i < tensor_size; i++) {
    auto array = static_cast<jintArray>(env->GetObjectArrayElement(dims, i));
    auto dim_size = static_cast<int>(env->GetArrayLength(array));
    jint *dim_data = env->GetIntArrayElements(array, nullptr);
    std::vector<int64_t> tensor_dims(dim_size);
    for (int j = 0; j < dim_size; j++) {
      tensor_dims[j] = dim_data[j];
    }
    c_dims.push_back(tensor_dims);
    env->ReleaseIntArrayElements(array, dim_data, JNI_ABORT);
    env->DeleteLocalRef(array);
  }
  auto ret = lite_model_ptr->Resize(c_inputs, c_dims);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_export(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                      jstring model_name, jint quantization_type,
                                                                      jboolean export_inference_only,
                                                                      jobjectArray tensorNames) {
  auto *model_pointer = reinterpret_cast<void *>(model_ptr);
  if (model_pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(model_pointer);
  auto model_path = env->GetStringUTFChars(model_name, JNI_FALSE);
  std::vector<std::string> output_tensor_names;
  if (tensorNames != NULL) {
    auto tensor_size = static_cast<int>(env->GetArrayLength(tensorNames));
    for (int i = 0; i < tensor_size; i++) {
      auto tensor_name = static_cast<jstring>(env->GetObjectArrayElement(tensorNames, i));
      output_tensor_names.emplace_back(env->GetStringUTFChars(tensor_name, JNI_FALSE));
      env->DeleteLocalRef(tensor_name);
    }
  }
  mindspore::QuantizationType quant_type;
  if (quantization_type >= static_cast<int>(mindspore::kNoQuant) &&
      quantization_type <= static_cast<int>(mindspore::kFullQuant)) {
    quant_type = static_cast<mindspore::QuantizationType>(quantization_type);
  } else {
    MS_LOGE("Invalid quantization_type : %d", quantization_type);
    return (jlong) nullptr;
  }
  auto ret = mindspore::Serialization::ExportModel(*lite_model_ptr, mindspore::kMindIR, model_path, quant_type,
                                                   export_inference_only, output_tensor_names);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_updateFeatureMaps(JNIEnv *env, jclass, jlong model_ptr,
                                                                                 jlongArray features) {
  auto size = static_cast<int>(env->GetArrayLength(features));
  jlong *input_data = env->GetLongArrayElements(features, nullptr);
  std::vector<mindspore::MSTensor> newFeatures;
  for (int i = 0; i < size; ++i) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOGE("Tensor pointer from java is nullptr");
      return (jboolean) false;
    }
    auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(tensor_pointer);
    newFeatures.emplace_back(*ms_tensor_ptr);
  }
  auto lite_model_ptr = reinterpret_cast<mindspore::Model *>(model_ptr);
  auto ret = lite_model_ptr->UpdateFeatureMaps(newFeatures);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getFeatureMaps(JNIEnv *env, jobject thiz,
                                                                             jlong model_ptr) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return ret;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto features = lite_model_ptr->GetFeatureMaps();
  for (auto &feature : features) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(feature);
    if (tensor_ptr == nullptr) {
      MS_LOGE("Make ms tensor failed");
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_setLearningRate(JNIEnv *env, jclass, jlong model_ptr,
                                                                               jfloat learning_rate) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto ret = lite_model_ptr->SetLearningRate(learning_rate);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_setupVirtualBatch(
  JNIEnv *env, jobject thiz, jlong model_ptr, jint virtual_batch_factor, jfloat learning_rate, jfloat momentum) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto ret = lite_model_ptr->SetupVirtualBatch(virtual_batch_factor, learning_rate, momentum);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_Model_free(JNIEnv *env, jobject thiz, jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  delete (lite_model_ptr);
}
