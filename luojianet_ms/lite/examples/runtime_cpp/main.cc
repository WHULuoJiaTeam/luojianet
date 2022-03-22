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

#include <iostream>
#include <cstring>
#include <random>
#include <fstream>
#include <thread>
#include <algorithm>
#include "include/api/allocator.h"
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"

std::string RealPath(const char *path) {
  const size_t max = 4096;
  if (path == nullptr) {
    std::cerr << "path is nullptr" << std::endl;
    return "";
  }
  if ((strlen(path)) >= max) {
    std::cerr << "path is too long" << std::endl;
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(max);
  if (resolved_path == nullptr) {
    std::cerr << "new resolved_path failed" << std::endl;
    return "";
  }
#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), path, 1024);
#else
  char *real_path = realpath(path, resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    std::cerr << "file path is not valid : " << path << std::endl;
    return "";
  }
  std::string res = resolved_path.get();
  return res;
}

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    std::cerr << "file is nullptr." << std::endl;
    return nullptr;
  }

  std::ifstream ifs(file, std::ifstream::in | std::ifstream::binary);
  if (!ifs.good()) {
    std::cerr << "file: " << file << " is not exist." << std::endl;
    return nullptr;
  }

  if (!ifs.is_open()) {
    std::cerr << "file: " << file << " open failed." << std::endl;
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    std::cerr << "malloc buf failed, file: " << file << std::endl;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  if (data == nullptr) {
    std::cerr << "data is nullptr." << std::endl;
    return;
  }
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&]() { return static_cast<T>(distribution(random_engine)); });
}

std::shared_ptr<luojianet_ms::CPUDeviceInfo> CreateCPUDeviceInfo() {
  auto device_info = std::make_shared<luojianet_ms::CPUDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New CPUDeviceInfo failed." << std::endl;
    return nullptr;
  }
  // Use float16 operator as priority.
  device_info->SetEnableFP16(true);
  return device_info;
}

std::shared_ptr<luojianet_ms::GPUDeviceInfo> CreateGPUDeviceInfo() {
  auto device_info = std::make_shared<luojianet_ms::GPUDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New GPUDeviceInfo failed." << std::endl;
    return nullptr;
  }
  // If GPU device info is set. The preferred backend is GPU, which means, if there is a GPU operator, it will run on
  // the GPU first, otherwise it will run on the CPU.
  // GPU use float16 operator as priority.
  device_info->SetEnableFP16(true);
  return device_info;
}

std::shared_ptr<luojianet_ms::KirinNPUDeviceInfo> CreateNPUDeviceInfo() {
  auto device_info = std::make_shared<luojianet_ms::KirinNPUDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New KirinNPUDeviceInfo failed." << std::endl;
    return nullptr;
  }
  device_info->SetFrequency(3);
  return device_info;
}

luojianet_ms::Status GetInputsAndSetData(luojianet_ms::Model *model) {
  auto inputs = model->GetInputs();
  // The model has only one input tensor.
  auto in_tensor = inputs.front();
  if (in_tensor == nullptr) {
    std::cerr << "Input tensor is nullptr" << std::endl;
    return luojianet_ms::kLiteNullptr;
  }
  auto input_data = in_tensor.MutableData();
  if (input_data == nullptr) {
    std::cerr << "MallocData for inTensor failed." << std::endl;
    return luojianet_ms::kLiteNullptr;
  }
  GenerateRandomData<float>(in_tensor.DataSize(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
  return luojianet_ms::kSuccess;
}

luojianet_ms::Status GetInputsByTensorNameAndSetData(luojianet_ms::Model *model) {
  auto in_tensor = model->GetInputByTensorName("graph_input-173");
  if (in_tensor == nullptr) {
    std::cerr << "Input tensor is nullptr" << std::endl;
    return luojianet_ms::kLiteNullptr;
  }
  auto input_data = in_tensor.MutableData();
  if (input_data == nullptr) {
    std::cerr << "MallocData for inTensor failed." << std::endl;
    return luojianet_ms::kLiteNullptr;
  }
  GenerateRandomData<float>(in_tensor.DataSize(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
  return luojianet_ms::kSuccess;
}

void GetOutputsByNodeName(luojianet_ms::Model *model) {
  // model has a output node named output_node_name_0.
  auto output_vec = model->GetOutputsByNodeName("Softmax-65");
  // output node named output_node_name_0 has only one output tensor.
  auto out_tensor = output_vec.front();
  if (out_tensor == nullptr) {
    std::cerr << "Output tensor is nullptr" << std::endl;
    return;
  }
  std::cout << "tensor size is:" << out_tensor.DataSize() << " tensor elements num is:" << out_tensor.ElementNum()
            << std::endl;
  // The model output data is float 32.
  if (out_tensor.DataType() != luojianet_ms::DataType::kNumberTypeFloat32) {
    std::cerr << "Output should in float32" << std::endl;
    return;
  }
  auto out_data = reinterpret_cast<float *>(out_tensor.MutableData());
  if (out_data == nullptr) {
    std::cerr << "Data of out_tensor is nullptr" << std::endl;
    return;
  }
  std::cout << "output data is:";
  for (int i = 0; i < out_tensor.ElementNum() && i < 10; i++) {
    std::cout << out_data[i] << " ";
  }
  std::cout << std::endl;
}

void GetOutputByTensorName(luojianet_ms::Model *model) {
  // We can use GetOutputTensorNames method to get all name of output tensor of model which is in order.
  auto tensor_names = model->GetOutputTensorNames();
  for (const auto &tensor_name : tensor_names) {
    auto out_tensor = model->GetOutputByTensorName(tensor_name);
    if (out_tensor == nullptr) {
      std::cerr << "Output tensor is nullptr" << std::endl;
      return;
    }
    std::cout << "tensor size is:" << out_tensor.DataSize() << " tensor elements num is:" << out_tensor.ElementNum()
              << std::endl;
    // The model output data is float 32.
    if (out_tensor.DataType() != luojianet_ms::DataType::kNumberTypeFloat32) {
      std::cerr << "Output should in float32" << std::endl;
      return;
    }
    auto out_data = reinterpret_cast<float *>(out_tensor.MutableData());
    if (out_data == nullptr) {
      std::cerr << "Data of out_tensor is nullptr" << std::endl;
      return;
    }
    std::cout << "output data is:";
    for (int i = 0; i < out_tensor.ElementNum() && i < 10; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
}

void GetOutputs(luojianet_ms::Model *model) {
  auto out_tensors = model->GetOutputs();
  for (auto out_tensor : out_tensors) {
    std::cout << "tensor name is:" << out_tensor.Name() << " tensor size is:" << out_tensor.DataSize()
              << " tensor elements num is:" << out_tensor.ElementNum() << std::endl;
    // The model output data is float 32.
    if (out_tensor.DataType() != luojianet_ms::DataType::kNumberTypeFloat32) {
      std::cerr << "Output should in float32" << std::endl;
      return;
    }
    auto out_data = reinterpret_cast<float *>(out_tensor.MutableData());
    if (out_data == nullptr) {
      std::cerr << "Data of out_tensor is nullptr" << std::endl;
      return;
    }
    std::cout << "output data is:";
    for (int i = 0; i < out_tensor.ElementNum() && i < 10; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
}

luojianet_ms::Model *CreateAndBuildModel(char *model_buf, size_t model_size) {
  // Create and init context, add CPU device info
  auto context = std::make_shared<luojianet_ms::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return nullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  // If you need to use GPU or NPU, you can refer to CreateGPUDeviceInfo() or CreateNPUDeviceInfo().
  auto cpu_device_info = CreateCPUDeviceInfo();
  if (cpu_device_info == nullptr) {
    std::cerr << "Create CPUDeviceInfo failed." << std::endl;
    return nullptr;
  }
  device_list.push_back(cpu_device_info);

  // Create model
  auto model = new (std::nothrow) luojianet_ms::Model();
  if (model == nullptr) {
    std::cerr << "New Model failed." << std::endl;
    return nullptr;
  }
  // Build model
  auto build_ret = model->Build(model_buf, model_size, luojianet_ms::kMindIR, context);
  if (build_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Build model failed." << std::endl;
    return nullptr;
  }
  return model;
}

luojianet_ms::Model *CreateAndBuildModelComplicated(char *model_buf, size_t size) {
  // Create and init context, add CPU device info
  auto context = std::make_shared<luojianet_ms::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return nullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  auto cpu_device_info = CreateCPUDeviceInfo();
  if (cpu_device_info == nullptr) {
    std::cerr << "Create CPUDeviceInfo failed." << std::endl;
    return nullptr;
  }
  device_list.push_back(cpu_device_info);

  // Load graph
  luojianet_ms::Graph graph;
  auto load_ret = luojianet_ms::Serialization::Load(model_buf, size, luojianet_ms::kMindIR, &graph);
  if (load_ret != luojianet_ms::kSuccess) {
    std::cerr << "Load graph failed." << std::endl;
    return nullptr;
  }

  // Create model
  auto model = new (std::nothrow) luojianet_ms::Model();
  if (model == nullptr) {
    std::cerr << "New Model failed." << std::endl;
    return nullptr;
  }
  // Build model
  luojianet_ms::GraphCell graph_cell(graph);
  auto build_ret = model->Build(graph_cell, context);
  if (build_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Build model failed." << std::endl;
    return nullptr;
  }
  return model;
}

luojianet_ms::Status ResizeInputsTensorShape(luojianet_ms::Model *model) {
  auto inputs = model->GetInputs();
  std::vector<int64_t> resize_shape = {1, 128, 128, 3};
  // Assume the model has only one input,resize input shape to [1, 128, 128, 3]
  std::vector<std::vector<int64_t>> new_shapes;
  new_shapes.push_back(resize_shape);
  return model->Resize(inputs, new_shapes);
}

int Run(const char *model_path) {
  // Read model file.
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Create and Build LuoJiaNet model.
  auto model = CreateAndBuildModel(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  auto generate_input_ret = GetInputsByTensorNameAndSetData(model);
  if (generate_input_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Set input data error " << generate_input_ret << std::endl;
    return -1;
  }

  auto inputs = model->GetInputs();
  auto outputs = model->GetOutputs();
  auto predict_ret = model->Predict(inputs, &outputs);
  if (predict_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(model);

  // Delete model.
  delete model;
  return 0;
}

int RunResize(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Create and Build LuoJiaNet model.
  auto model = CreateAndBuildModel(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }

  // Resize inputs tensor shape.
  auto resize_ret = ResizeInputsTensorShape(model);
  if (resize_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Resize input tensor shape error." << resize_ret << std::endl;
    return -1;
  }

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  auto generate_input_ret = GetInputsByTensorNameAndSetData(model);
  if (generate_input_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Set input data error " << generate_input_ret << std::endl;
    return -1;
  }

  auto inputs = model->GetInputs();
  auto outputs = model->GetOutputs();
  auto predict_ret = model->Predict(inputs, &outputs);
  if (predict_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(model);

  // Delete model.
  delete model;
  return 0;
}

int RunCreateModelComplicated(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Create and Build LuoJiaNet model.
  auto model = CreateAndBuildModelComplicated(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  auto generate_input_ret = GetInputsByTensorNameAndSetData(model);
  if (generate_input_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Set input data error " << generate_input_ret << std::endl;
    return -1;
  }

  auto inputs = model->GetInputs();
  auto outputs = model->GetOutputs();
  auto predict_ret = model->Predict(inputs, &outputs);
  if (predict_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(model);

  // Delete model.
  delete model;
  return 0;
}

int RunModelParallel(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Create and Build LuoJiaNet model.
  auto model1 = CreateAndBuildModel(model_buf, size);
  auto model2 = CreateAndBuildModel(model_buf, size);
  delete[](model_buf);
  if (model1 == nullptr || model2 == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }

  std::thread thread1([&]() {
    auto generate_input_ret = GetInputsByTensorNameAndSetData(model1);
    if (generate_input_ret != luojianet_ms::kSuccess) {
      std::cerr << "Model1 set input data error " << generate_input_ret << std::endl;
      return -1;
    }

    auto inputs = model1->GetInputs();
    auto outputs = model1->GetOutputs();
    auto predict_ret = model1->Predict(inputs, &outputs);
    if (predict_ret != luojianet_ms::kSuccess) {
      std::cerr << "Model1 predict error " << predict_ret << std::endl;
      return -1;
    }
    std::cout << "Model1 predict success" << std::endl;
    return 0;
  });

  std::thread thread2([&]() {
    auto generate_input_ret = GetInputsByTensorNameAndSetData(model2);
    if (generate_input_ret != luojianet_ms::kSuccess) {
      std::cerr << "Model2 set input data error " << generate_input_ret << std::endl;
      return -1;
    }

    auto inputs = model2->GetInputs();
    auto outputs = model2->GetOutputs();
    auto predict_ret = model2->Predict(inputs, &outputs);
    if (predict_ret != luojianet_ms::kSuccess) {
      std::cerr << "Model2 predict error " << predict_ret << std::endl;
      return -1;
    }
    std::cout << "Model2 predict success" << std::endl;
    return 0;
  });

  thread1.join();
  thread2.join();

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(model1);
  GetOutputsByNodeName(model2);

  // Delete model.
  delete model1;
  delete model2;
  return 0;
}

int RunWithSharedMemoryPool(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  auto context1 = std::make_shared<luojianet_ms::Context>();
  if (context1 == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return -1;
  }
  auto &device_list1 = context1->MutableDeviceInfo();
  auto device_info1 = CreateCPUDeviceInfo();
  if (device_info1 == nullptr) {
    std::cerr << "Create CPUDeviceInfo failed." << std::endl;
    return -1;
  }
  device_list1.push_back(device_info1);

  auto model1 = new (std::nothrow) luojianet_ms::Model();
  if (model1 == nullptr) {
    delete[](model_buf);
    std::cerr << "New Model failed." << std::endl;
    return -1;
  }
  auto build_ret = model1->Build(model_buf, size, luojianet_ms::kMindIR, context1);
  if (build_ret != luojianet_ms::kSuccess) {
    delete[](model_buf);
    delete model1;
    std::cerr << "Build model failed." << std::endl;
    return -1;
  }

  auto context2 = std::make_shared<luojianet_ms::Context>();
  if (context2 == nullptr) {
    delete[](model_buf);
    delete model1;
    std::cerr << "New context failed." << std::endl;
    return -1;
  }
  auto &device_list2 = context2->MutableDeviceInfo();
  auto device_info2 = CreateCPUDeviceInfo();
  if (device_info2 == nullptr) {
    delete[](model_buf);
    delete model1;
    std::cerr << "Create CPUDeviceInfo failed." << std::endl;
    return -1;
  }
  // Use the same allocator to share the memory pool.
  device_info2->SetAllocator(device_info1->GetAllocator());
  device_list2.push_back(device_info2);

  auto model2 = new (std::nothrow) luojianet_ms::Model();
  if (model2 == nullptr) {
    delete[](model_buf);
    delete model1;
    std::cerr << "New Model failed." << std::endl;
    return -1;
  }
  build_ret = model2->Build(model_buf, size, luojianet_ms::kMindIR, context2);
  delete[](model_buf);
  if (build_ret != luojianet_ms::kSuccess) {
    delete model1;
    delete model2;
    std::cerr << "Build model failed." << std::endl;
    return -1;
  }

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  GetInputsByTensorNameAndSetData(model1);
  GetInputsByTensorNameAndSetData(model2);

  auto inputs1 = model1->GetInputs();
  auto outputs1 = model1->GetOutputs();
  auto predict_ret = model1->Predict(inputs1, &outputs1);
  if (predict_ret != luojianet_ms::kSuccess) {
    delete model1;
    delete model2;
    std::cerr << "Inference error " << predict_ret << std::endl;
    return -1;
  }

  auto inputs2 = model2->GetInputs();
  auto outputs2 = model2->GetOutputs();
  predict_ret = model2->Predict(inputs2, &outputs2);
  if (predict_ret != luojianet_ms::kSuccess) {
    delete model1;
    delete model2;
    std::cerr << "Inference error " << predict_ret << std::endl;
    return -1;
  }

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(model1);
  GetOutputsByNodeName(model2);

  // Delete model.
  delete model1;
  delete model2;
  return 0;
}

int RunCallback(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Create and Build LuoJiaNet model.
  auto model = CreateAndBuildModel(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    delete model;
    std::cerr << "Create model failed." << std::endl;
    return -1;
  }

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  auto generate_input_ret = GetInputsByTensorNameAndSetData(model);
  if (generate_input_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Set input data error " << generate_input_ret << std::endl;
    return -1;
  }

  // Definition of callback function before forwarding operator.
  auto before_call_back = [](const std::vector<luojianet_ms::MSTensor> &before_inputs,
                             const std::vector<luojianet_ms::MSTensor> &before_outputs,
                             const luojianet_ms::MSCallBackParam &call_param) {
    std::cout << "Before forwarding " << call_param.node_name << " " << call_param.node_type << std::endl;
    return true;
  };
  // Definition of callback function after forwarding operator.
  auto after_call_back = [](const std::vector<luojianet_ms::MSTensor> &after_inputs,
                            const std::vector<luojianet_ms::MSTensor> &after_outputs,
                            const luojianet_ms::MSCallBackParam &call_param) {
    std::cout << "After forwarding " << call_param.node_name << " " << call_param.node_type << std::endl;
    return true;
  };

  auto inputs = model->GetInputs();
  auto outputs = model->GetOutputs();
  auto predict_ret = model->Predict(inputs, &outputs, before_call_back, after_call_back);
  if (predict_ret != luojianet_ms::kSuccess) {
    delete model;
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(model);

  // Delete model.
  delete model;
  return 0;
}

int main(int argc, const char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./runtime_cpp model_path Option" << std::endl;
    std::cerr << "Example: ./runtime_cpp ../model/mobilenetv2.ms 0" << std::endl;
    std::cerr << "When your Option is 0, you will run LuoJiaNet Lite predict." << std::endl;
    std::cerr << "When your Option is 1, you will run LuoJiaNet Lite predict with resize." << std::endl;
    std::cerr << "When your Option is 2, you will run LuoJiaNet Lite predict with complicated API." << std::endl;
    std::cerr << "When your Option is 3, you will run LuoJiaNet Lite predict with model parallel." << std::endl;
    std::cerr << "When your Option is 4, you will run LuoJiaNet Lite predict with shared memory pool." << std::endl;
    std::cerr << "When your Option is 5, you will run LuoJiaNet Lite predict with callback." << std::endl;
    return -1;
  }
  std::string version = luojianet_ms::Version();
  std::cout << "LuoJiaNet Lite Version is " << version << std::endl;
  auto model_path = RealPath(argv[1]);
  if (model_path.empty()) {
    std::cerr << "model path " << argv[1] << " is invalid.";
    return -1;
  }
  auto flag = argv[2];
  if (strcmp(flag, "0") == 0) {
    return Run(model_path.c_str());
  } else if (strcmp(flag, "1") == 0) {
    return RunResize(model_path.c_str());
  } else if (strcmp(flag, "2") == 0) {
    return RunCreateModelComplicated(model_path.c_str());
  } else if (strcmp(flag, "3") == 0) {
    return RunModelParallel(model_path.c_str());
  } else if (strcmp(flag, "4") == 0) {
    return RunWithSharedMemoryPool(model_path.c_str());
  } else if (strcmp(flag, "5") == 0) {
    return RunCallback(model_path.c_str());
  } else {
    std::cerr << "Unsupported Flag " << flag << std::endl;
    return -1;
  }
}
