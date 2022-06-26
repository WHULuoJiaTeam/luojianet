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

#include "securec/include/securec.h"
#include "include/ms_tensor.h"
#include "src/tensor.h"

namespace luojianet_ms {
namespace tensor {
tensor::MSTensor *tensor::MSTensor::CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape,
                                                 const void *data, size_t data_len) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }

  size_t shape_size = 1;
  if (shape.empty()) {
    shape_size = 0;
  } else {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        delete tensor;
        return nullptr;
      }
      shape_size *= static_cast<size_t>(shape[i]);
    }
  }
  auto data_type_size = lite::DataTypeSize(type);
  if (data_type_size == 0) {
    MS_LOG(ERROR) << "not support create this type: " << type;
    delete tensor;
    return nullptr;
  }

  if (data == nullptr && data_len != 0) {
    MS_LOG(ERROR) << "shape, data type and data len not match.";
    delete tensor;
    return nullptr;
  }

  if (data != nullptr && data_len != shape_size * data_type_size) {
    MS_LOG(ERROR) << "shape, data type and data len not match.";
    delete tensor;
    return nullptr;
  }
  tensor->set_data(const_cast<void *>(data));
  tensor->set_shape(shape);
  tensor->set_tensor_name(name);
  tensor->set_data_type(type);
  return tensor;
}

tensor::MSTensor *tensor::MSTensor::CreateTensorByDeepCopy(const std::string &name, TypeId type,
                                                           const std::vector<int> &shape, const void *data,
                                                           size_t data_len) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }

  auto data_type_size = lite::DataTypeSize(type);
  if (data_type_size == 0) {
    MS_LOG(ERROR) << "not support create this type: " << type;
    delete tensor;
    return nullptr;
  }

  if (data_len < 0 || data_len > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "data length is invalid.";
    delete tensor;
    return nullptr;
  } else if (data_len == 0 && data != nullptr) {
    MS_LOG(ERROR) << "data length and data are not match.";
    delete tensor;
    return nullptr;
  } else if (data_len == 0 && data == nullptr) {
    tensor->set_data(const_cast<void *>(data));
  } else {
    void *new_data = malloc(data_len);
    if (new_data == nullptr) {
      MS_LOG(ERROR) << "Failed to malloc data.";
      delete tensor;
      return nullptr;
    }
    if (data != nullptr) {
      (void)memcpy(new_data, data, data_len);
    }
    tensor->set_data(const_cast<void *>(new_data));
  }

  size_t shape_size = 1;
  if (shape.empty()) {
    shape_size = 0;
  } else {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        delete tensor;
        return nullptr;
      }
      shape_size *= static_cast<size_t>(shape[i]);
    }
  }
  if (data_len != shape_size * data_type_size) {
    std::vector<int> truncate_shape = {static_cast<int>(data_len)};
    tensor->set_shape(truncate_shape);
  } else {
    tensor->set_shape(shape);
  }
  tensor->set_tensor_name(name);
  tensor->set_data_type(type);
  return tensor;
}
}  // namespace tensor
}  // namespace luojianet_ms
