
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

#ifndef LUOJIANET_MS_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_
#define LUOJIANET_MS_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_

#include "include/ms_tensor.h"
#include "ir/format.h"

namespace luojianet_ms {
namespace lite {
struct LiteQuantParam {
  double scale;
  int32_t zeroPoint;
  float var_corr{1};
  float mean_corr{0};
  bool inited;
  Vector<float> clusters{};
  int bitNum;
  int roundType;
  int multiplier;
  int dstDtype;
};

class MTensor : public luojianet_ms::tensor::MSTensor {
 public:
  MTensor() = default;
  MTensor(String name, TypeId type, Vector<int> shape) : tensor_name_(name), data_type_(type), shape_(shape) {}
  ~MTensor() override;

  void set_allocator(luojianet_ms::Allocator *allocator) override {}
  luojianet_ms::Allocator *allocator() const override { return nullptr; }
  TypeId data_type() const override { return data_type_; }
  void set_data_type(TypeId data_type) override { data_type_ = data_type; }
  void set_format(luojianet_ms::Format format) override {}
  luojianet_ms::Format format() const override { return luojianet_ms::NHWC; }
  Vector<int> shape() const override { return shape_; }
  void set_shape(const Vector<int> &shape) override { shape_ = shape; }
  int ElementsNum() const override;
  size_t Size() const override;
  String tensor_name() const override { return tensor_name_; }
  void set_tensor_name(const String &name) override { tensor_name_ = name; }
  void *MutableData() override;
  void *data() override { return data_; }
  void set_data(void *data) override { data_ = data; }
  Vector<LiteQuantParam> quant_params() const override { return this->quant_params_; }
  void set_quant_params(const Vector<LiteQuantParam> quant_params) override { this->quant_params_ = quant_params; }

 private:
  String tensor_name_;
  TypeId data_type_;
  Vector<int> shape_;
  void *data_ = nullptr;
  Vector<LiteQuantParam> quant_params_;
};
}  // namespace lite
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_
