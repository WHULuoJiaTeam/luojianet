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

#include "ir/tensor.h"

#include <iomanip>
#include <functional>
#include <utility>
#include <algorithm>
#include "abstract/utils.h"
#include "abstract/abstract_value.h"
#include "base/complex_storage.h"

namespace mindspore {
namespace tensor {
constexpr auto kEllipsis = "...";
constexpr auto kThreshold = 6;

constexpr auto kThreshold1DFloat = kThreshold * 2;
constexpr auto kThreshold1DInt = kThreshold * 4;
constexpr auto kThreshold1DBool = kThreshold * 2;

static std::string MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return "T" + std::to_string(last_id.fetch_add(1, std::memory_order_relaxed));
}

static TypeId TypeIdOf(const TypePtr &data_type, TypeId defaultTypeId) {
  return data_type ? data_type->type_id() : defaultTypeId;
}

static size_t SizeOf(const ShapeVector &shape) {
  int64_t data_size = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      // For dynamic shape which has negative dimensions, data size should be zero.
      return 0;
    }
    data_size *= dim;
  }
  return static_cast<size_t>(data_size);
}

static std::string ShapeToString(const ShapeVector &shape) {
  std::string str = "[";
  const size_t count = shape.size();
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      str.append(", ");
    }
    str.append(std::to_string(shape[i]));
  }
  return str.append("]");
}

template <typename T, typename U>
std::unique_ptr<T[]> NewData(const U *input, size_t size) {
  if (input == nullptr || size == 0) {
    return nullptr;
  }
  if (size > INT32_MAX) {
    MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << size * sizeof(T);
  }

  auto data = std::make_unique<T[]>(size);
  if constexpr (!std::is_same<T, U>::value &&
                (std::is_same<T, float16>::value || std::is_same<U, float16>::value ||
                 std::is_same<T, ComplexStorage<float>>::value || std::is_same<U, ComplexStorage<float>>::value ||
                 std::is_same<T, ComplexStorage<double>>::value || std::is_same<U, ComplexStorage<double>>::value)) {
    // Because float16 do not support implicit cast from/to other types,
    // We can not use std::copy() on array of float16, use a loop here.
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<T>(input[i]);
    }
  } else {
    // otherwise, use std::copy for better performance.
    std::copy(input, input + size, data.get());
  }
  return data;
}

template <typename T, typename Scalar>
std::unique_ptr<T[]> NewData(Scalar scalar) {
  auto data = std::make_unique<T[]>(1);
  data[0] = static_cast<T>(scalar);
  return data;
}

template <typename T>
std::unique_ptr<T[]> CopyData(const ShapeVector &shape, void *const data, TypeId data_type) {
  const size_t size = SizeOf(shape);
  switch (data_type) {
    case kNumberTypeBool: {
      auto buf = static_cast<bool *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt8: {
      auto buf = static_cast<uint8_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt8: {
      auto buf = static_cast<int8_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt16: {
      auto buf = static_cast<int16_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt32: {
      auto buf = static_cast<int32_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt64: {
      auto buf = static_cast<int64_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt16: {
      auto buf = static_cast<uint16_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt32: {
      auto buf = static_cast<uint32_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt64: {
      auto buf = static_cast<uint64_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat16: {
      auto buf = static_cast<float16 *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat32: {
      auto buf = static_cast<float *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat64: {
      auto buf = static_cast<double *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeComplex64: {
      auto buf = static_cast<ComplexStorage<float> *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeComplex128: {
      auto buf = static_cast<ComplexStorage<double> *>(data);
      return NewData<T>(buf, size);
    }
    case kObjectTypeString: {
      auto buf = static_cast<uint8_t *>(data);
      return NewData<T>(buf, size);
    }
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
}

template <typename T>
std::unique_ptr<T[]> CopyData(const ShapeVector &shape, void *const data, size_t data_len) {
  size_t size = SizeOf(shape);
  if (size * sizeof(T) != data_len) {
    MS_LOG(EXCEPTION) << "Incorrect tensor input data length  " << data_len << ", expect " << size * sizeof(T)
                      << " item size " << sizeof(T);
  }
  auto buf = static_cast<T *>(data);
  return NewData<T>(buf, size);
}

// Tensor data implementation.
template <typename T>
class TensorDataImpl : public TensorData {
 public:
  explicit TensorDataImpl(const ShapeVector &shape) : ndim_(shape.size()), data_size_(SizeOf(shape)) {}
  ~TensorDataImpl() = default;

  TensorDataImpl(const ShapeVector &shape, void *data, size_t data_len)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_len)) {}

  TensorDataImpl(const ShapeVector &shape, void *data, TypeId data_type)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_type)) {}

  template <typename U>
  TensorDataImpl(const ShapeVector &shape, const U *input, size_t size)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(input, size)) {}

  template <typename Scalar>
  TensorDataImpl(const ShapeVector &shape, Scalar scalar)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(scalar)) {}

  ssize_t size() const override { return static_cast<ssize_t>(data_size_); }

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(ndim_); }

  void *data() override {
    if (data_ == nullptr) {
      if (data_size_ > INT32_MAX) {
        MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << data_size_ * sizeof(T);
      }
      // Lazy allocation.
      data_ = std::make_unique<T[]>(data_size_);
    }
    return data_.get();
  }

  const void *const_data() const override {
    // May return nullptr if data not initialized.
    return data_.get();
  }

  virtual bool equals(const TensorDataImpl<T> &other) const {
    auto ptr = &other;
    if (ptr == this) {
      return true;
    }
    if (data_ == nullptr || ptr->data_ == nullptr) {
      return false;
    }
    return (ndim_ == ptr->ndim_) && (data_size_ == ptr->data_size_) &&
           std::equal(data_.get(), data_.get() + data_size_, ptr->data_.get());
  }

  bool equals(const TensorData &other) const override {
    // Not same type, compare data byte by byte.
    return TensorData::equals(other);
  }

  std::string ToString(const TypeId type, const ShapeVector &shape, bool use_comma) const override {
    constexpr auto valid =
      std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value ||
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, uint16_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value ||
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value ||
      std::is_same<T, ComplexStorage<float>>::value || std::is_same<T, ComplexStorage<double>>::value;
    static_assert(valid, "Type is invalid");
    if (data_size_ == 0) {
      return "";
    }
    if (data_ == nullptr) {
      return "<uninitialized>";
    }

    std::ostringstream ss;
    if (data_size_ == 1 && ndim_ == 0) {  // Scalar
      OutputDataString(ss, 0, 0, 1, false, 0);
      return ss.str();
    }

    int num_width = 0;
    ssize_t cursor = 0;
    SummaryStringRecursive(ss, shape, &cursor, 0, use_comma, &num_width);
    return ProcessPlaceholder(ss, num_width);
  }

 private:
  void OutputFloatDataString(std::ostringstream &ss, bool isScalar, const T &value) const {
    if (isScalar) {
      ss << value;
    } else {
      // The placeholder of float16 is fixed at 11, while float/double is fixed at 15.
      const int width = std::is_same<T, float16>::value ? 11 : 15;
      // The printing precision of float16 is fixed at 4, while float/double is fixed at 8.
      const int precision = std::is_same<T, float16>::value ? 4 : 8;
      ss << std::setw(width) << std::setprecision(precision) << std::setiosflags(std::ios::scientific | std::ios::right)
         << value;
    }
  }

  void OutputBoolDataString(std::ostringstream &ss, bool isScalar, const T &value) const {
    if (isScalar) {
      ss << (value ? "True" : "False");
    } else {
      constexpr int bool_max_width = sizeof("False") - 1;
      ss << std::setw(bool_max_width) << std::setiosflags(std::ios::right) << (value ? "True" : "False");
    }
  }

  void OutputOtherDataString(std::ostringstream &ss, bool isScalar, const T &value, int *max_width) const {
    if (isScalar) {
      ss << value;
    } else {
      // Add a padding string before the number, such as "###123", for subsequent replacement.
      const int width = GetNumLength(value);
      *max_width = std::max(*max_width, width);
      std::string pad(width, '#');
      ss << pad;
      if constexpr (std::is_same<T, uint8_t>::value) {
        ss << static_cast<uint16_t>(value);
      } else if constexpr (std::is_same<T, int8_t>::value) {
        ss << static_cast<int16_t>(value);
      } else {
        ss << value;
      }
    }
  }

  void OutputDataString(std::ostringstream &ss, ssize_t cursor, ssize_t start, ssize_t end, bool use_comma,
                        int *max_width) const {
    const bool isScalar = ndim_ == 0 && end - start == 1;
    constexpr auto isBool = std::is_same<T, bool>::value;
    constexpr auto isFloat =
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value;
    constexpr auto isComplex =
      std::is_same<T, ComplexStorage<float>>::value || std::is_same<T, ComplexStorage<double>>::value;
    constexpr int linefeedThreshold = isFloat ? kThreshold1DFloat : (isBool ? kThreshold1DBool : kThreshold1DInt);
    for (ssize_t i = start; i < end && (cursor + i) < static_cast<ssize_t>(data_size_); i++) {
      const auto value = data_[cursor + i];
      if constexpr (isComplex) {
        ss << value;
      } else if constexpr (isFloat) {
        OutputFloatDataString(ss, isScalar, value);
      } else if (isBool) {
        OutputBoolDataString(ss, isScalar, value);
      } else {
        OutputOtherDataString(ss, isScalar, value, max_width);
      }
      if (!isScalar && i != end - 1) {
        if (use_comma) {
          ss << ',';
        }
        ss << ' ';
      }
      if (!isScalar && ndim_ == 1 && (i + 1) % linefeedThreshold == 0) {
        // Add a line feed every {threshold of type} for 1D tensor.
        ss << '\n' << ' ';
      }
    }
  }

  void SummaryStringRecursive(std::ostringstream &ss, const ShapeVector &shape, ssize_t *cursor, ssize_t depth,
                              bool use_comma, int *max_width) const {
    if (depth >= static_cast<ssize_t>(ndim_)) {
      return;
    }
    ss << '[';
    if (depth == static_cast<ssize_t>(ndim_) - 1) {  // Bottom dimension
      ssize_t num = shape[depth];
      if (num > kThreshold && ndim_ > 1) {
        OutputDataString(ss, *cursor, 0, kThreshold >> 1, use_comma, max_width);
        ss << ' ' << kEllipsis << ' ';
        OutputDataString(ss, *cursor, num - (kThreshold >> 1), num, use_comma, max_width);
      } else {
        OutputDataString(ss, *cursor, 0, num, use_comma, max_width);
      }
      *cursor += num;
    } else {  // Middle dimension
      ssize_t num = shape[depth];
      // Handle the first half.
      for (ssize_t i = 0; i < std::min(static_cast<ssize_t>(kThreshold >> 1), num); i++) {
        if (i > 0) {
          if (use_comma) {
            ss << ',';
          }
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
        }
        SummaryStringRecursive(ss, shape, cursor, depth + 1, use_comma, max_width);
      }
      // Handle the ignored part.
      if (num > kThreshold) {
        if (use_comma) {
          ss << ',';
        }
        ss << '\n';
        ss << std::setw(depth + 1) << ' ';  // Add the indent.
        ss << kEllipsis;
        // Ignored at this layer.
        ssize_t ignored = shape[depth + 1];
        const size_t offset = 2;
        for (ssize_t i = depth + offset; i < static_cast<ssize_t>(ndim_); i++) {
          ignored *= shape[i];
        }
        // Multiple with ignored layers number.
        ignored *= num - kThreshold;
        *cursor += ignored;
      }
      // Handle the second half.
      if (num > (kThreshold >> 1)) {
        ssize_t iter_times =
          std::min(static_cast<ssize_t>(num - (kThreshold >> 1)), static_cast<ssize_t>(kThreshold >> 1));
        for (ssize_t i = 0; i < iter_times; i++) {
          if (use_comma && (i != 0 || num <= kThreshold)) {  // Not just after ignored part || Not handle ignored part
            ss << ',';
          }
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
          SummaryStringRecursive(ss, shape, cursor, depth + 1, use_comma, max_width);
        }
      }
    }
    ss << ']';
  }

  std::string ProcessPlaceholder(const std::ostringstream &ss, int max_width) const {
    std::string str = ss.str();
    if constexpr (std::is_same<T, bool>::value || std::is_same<T, float16>::value || std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
      return str;
    }
    // Replace # with placeholder.
    size_t index = str.find('#');
    while (index != std::string::npos) {
      size_t pos = index;
      while (str[pos] == '#') {
        pos++;
      }
      size_t len = pos - index;
      std::string space(max_width - SizeToInt(len), ' ');
      str = str.replace(index, len, space);
      index = str.find('#', index);
    }
    return str;
  }

  int GetNumLength(const T &num) const {
    T value = num;
    int count = 0;
    if (value <= 0) {  // Add the length of '-' when value < 0.
      count++;
    }
    while (value != 0) {
      value /= 10;
      count++;
    }
    return count;
  }

  size_t ndim_{0};
  size_t data_size_{0};
  std::unique_ptr<T[]> data_;
};

template <typename... Args>
TensorDataPtr MakeTensorData(TypeId data_type, const ShapeVector &shape, const Args... args) {
  switch (data_type) {
    case kNumberTypeBool:
      return std::make_shared<TensorDataImpl<bool>>(shape, args...);
    case kNumberTypeUInt8:
      return std::make_shared<TensorDataImpl<uint8_t>>(shape, args...);
    case kNumberTypeInt8:
      return std::make_shared<TensorDataImpl<int8_t>>(shape, args...);
    case kNumberTypeInt16:
      return std::make_shared<TensorDataImpl<int16_t>>(shape, args...);
    case kNumberTypeInt32:
      return std::make_shared<TensorDataImpl<int32_t>>(shape, args...);
    case kNumberTypeInt64:
      return std::make_shared<TensorDataImpl<int64_t>>(shape, args...);
    case kNumberTypeUInt16:
      return std::make_shared<TensorDataImpl<uint16_t>>(shape, args...);
    case kNumberTypeUInt32:
      return std::make_shared<TensorDataImpl<uint32_t>>(shape, args...);
    case kNumberTypeUInt64:
      return std::make_shared<TensorDataImpl<uint64_t>>(shape, args...);
    case kNumberTypeFloat16:
      return std::make_shared<TensorDataImpl<float16>>(shape, args...);
    case kNumberTypeFloat:
      return std::make_shared<TensorDataImpl<float>>(shape, args...);
    case kNumberTypeFloat32:
      return std::make_shared<TensorDataImpl<float>>(shape, args...);
    case kNumberTypeFloat64:
      return std::make_shared<TensorDataImpl<double>>(shape, args...);
    case kNumberTypeComplex64:
      return std::make_shared<TensorDataImpl<ComplexStorage<float>>>(shape, args...);
    case kNumberTypeComplex128:
      return std::make_shared<TensorDataImpl<ComplexStorage<double>>>(shape, args...);
    case kObjectTypeString:
      return std::make_shared<TensorDataImpl<uint8_t>>(shape, args...);
    case kObjectTypeTensorType:
      return std::make_shared<TensorDataImpl<int>>(shape, args...);
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
}

Tensor::Tensor(const Tensor &tensor)
    : MetaTensor(tensor),
      init_flag_(tensor.init_flag_),
      is_forward_output_(tensor.is_forward_output_),
      data_(tensor.data_),
      id_(tensor.id_),
      event_(tensor.event_),
      need_wait_(tensor.need_wait_),
      sync_status_(tensor.sync_status_),
      device_sync_(tensor.device_sync_),
      need_release_device_mem_(tensor.need_release_device_mem_),
      cache_enable_(tensor.cache_enable_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      padding_type_(tensor.padding_type()),
      device_event_(tensor.device_event_),
      lazy_callback_(tensor.lazy_callback_) {}

Tensor::Tensor(const Tensor &tensor, TypeId data_type)
    : MetaTensor(data_type, tensor.shape_),
      init_flag_(tensor.init_flag_),
      is_forward_output_(tensor.is_forward_output_),
      data_(MakeTensorData(data_type, tensor.shape_, tensor.data_->data(), tensor.data_type_)),
      id_(tensor.id_),
      event_(tensor.event_),
      need_wait_(tensor.need_wait_),
      sync_status_(tensor.sync_status_),
      device_sync_(tensor.device_sync_),
      need_release_device_mem_(tensor.need_release_device_mem_),
      cache_enable_(tensor.cache_enable_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      padding_type_(tensor.padding_type()),
      device_event_(tensor.device_event_),
      lazy_callback_(tensor.lazy_callback_) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, TensorDataPtr data)
    : MetaTensor(data_type, shape), data_(std::move(data)), id_(MakeId()) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape, data, data_len)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape, data, src_data_type)) {}

Tensor::Tensor(const std::vector<int64_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())),
      id_(MakeId()) {}

Tensor::Tensor(const std::vector<double> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())),
      id_(MakeId()) {}

Tensor::Tensor(int64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {}),
      data_(MakeTensorData(data_type_, {}, input)),
      id_(MakeId()) {}

Tensor::Tensor(double input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      data_(MakeTensorData(data_type_, {}, input)),
      id_(MakeId()) {}

Tensor::Tensor(uint64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt64), {}),
      data_(MakeTensorData(data_type_, {}, input)),
      id_(MakeId()) {}

Tensor::Tensor(bool input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeBool), {}),
      data_(MakeTensorData(data_type_, {}, input)),
      id_(MakeId()) {}

bool Tensor::operator==(const Tensor &tensor) const {
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_ == tensor.data_));
}

bool Tensor::ValueEqual(const Tensor &tensor) const {
  if (is_parameter_ != tensor.is_parameter_) {
    return false;
  }
  if (is_parameter_ && param_info_->name() != tensor.param_info_->name()) {
    return false;
  }
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_->equals(*tensor.data_)));
}

void Tensor::ExecuteLazyTask() const {
  if (lazy_callback_ != nullptr) {
    lazy_callback_();
  }
}

// assign value to this tensor
Tensor &Tensor::AssignValue(const Tensor &tensor) {
  if (this != &tensor) {
    lazy_callback_ = tensor.lazy_callback_;
    ExecuteLazyTask();
    MetaTensor::operator=(tensor);
    device_sync_ = tensor.device_sync_;
    is_forward_output_ = tensor.is_forward_output_;
    need_release_device_mem_ = tensor.need_release_device_mem_;
    data_ = tensor.data_;
    id_ = tensor.id_;
    event_ = tensor.event_;
    need_wait_ = tensor.need_wait_;
    sync_status_ = tensor.sync_status_;
    padding_type_ = tensor.padding_type_;
    device_event_ = tensor.device_event_;
  }
  return *this;
}

abstract::AbstractBasePtr Tensor::ToAbstract() {
  auto tens = shared_from_base<Tensor>();
  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  auto tensor_shape = tens->shape();
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  // if is parameter always no value.
  if (is_parameter_) {
    auto param_name = param_info_->name();
    auto ref_key = std::make_shared<RefKey>(param_name);
    auto abs_ref_key = ref_key->ToAbstract();
    abs_tensor = std::make_shared<abstract::AbstractRef>(abs_ref_key, abs_tensor);
  } else {
    abs_tensor->set_value(shared_from_base<Tensor>());
  }
  return abs_tensor;
}

std::string Tensor::GetShapeAndDataTypeInfo() const {
  std::ostringstream buf;
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  return buf.str();
}

std::string Tensor::ToStringInternal(size_t limit_size) const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", value=";
  if (limit_size == 0 || DataSize() < limit_size) {
    // Only print data for small tensor.
    buf << ((data().ndim() > 1) ? "\n" : "") << data().ToString(data_type_, shape_, false);
  } else {
    buf << "[...]";
  }
  if (is_parameter_) {
    buf << ", name=" << param_info_->name();
  }
  buf << ")";
  return buf.str();
}

std::string Tensor::ToString() const {
  constexpr size_t small_tensor_size = 30;
  return ToStringInternal(small_tensor_size);
}

std::string Tensor::ToStringNoLimit() const { return ToStringInternal(0); }

std::string Tensor::ToStringRepr() const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", value=" << ((data().ndim() > 1) ? '\n' : ' ') << data().ToString(data_type_, shape_, true) << ')';
  return buf.str();
}

void Tensor::data_sync(bool need_wait) const {
  if (need_wait) {
    ExecuteLazyTask();
    Wait();
  }
  if (device_sync_ == nullptr) {
    return;
  }
  std::vector<size_t> shape_tmp;
  (void)std::transform(shape().begin(), shape().end(), std::back_inserter(shape_tmp), IntToSize);
  auto size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(data_type());
  auto address = device_sync_;
  if (size != 0 && !address->SyncDeviceToHost(shape(), size, data_type(), data_c())) {
    MS_LOG(EXCEPTION) << "SyncDeviceToHost failed.";
  }
  sync_status_ = kNeedSyncHostToDevice;
}

TypeId Tensor::set_data_type(const TypeId data_type) {
  if (data_type != data_type_) {
    data_ = MakeTensorData(data_type, shape_, data_->data(), data_type_);
    return MetaTensor::set_data_type(data_type);
  }
  return data_type;
}

CSRTensor::CSRTensor(const TensorPtr indptr, const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
    : MetaSparseTensor(values->data_type(), shape), indptr_(indptr), indices_(indices), values_(values) {}

bool CSRTensor::operator==(const CSRTensor &csr_tensor) const { return (&csr_tensor == this); }

std::string CSRTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(values_);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indptr_);
  auto dtype = values_->Dtype();
  buf << "CSRTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", indptr=";
  buf << indptr_->ToString() << ", indices=" << indices_->ToString() << ", values=";
  buf << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr CSRTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  auto abs_csr_tensor = std::make_shared<abstract::AbstractCSRTensor>(dtype, shape_);

  abs_csr_tensor->set_indptr(indptr_->ToAbstract()->cast<abstract::AbstractTensorPtr>());
  abs_csr_tensor->set_indices(indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>());
  abs_csr_tensor->set_values(values_->ToAbstract()->cast<abstract::AbstractTensorPtr>());

  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  abs_csr_tensor->set_dense_shape(std::make_shared<abstract::AbstractTuple>(abstract_shape));

  return abs_csr_tensor;
}

const size_t CSRTensor::GetSizeAt(size_t index) const {
  if (index == kIndptrIdx) {
    MS_EXCEPTION_IF_NULL(indptr_);
    return indptr_->data().nbytes();
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_->data().nbytes();
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_->data().nbytes();
  } else if (index >= kIndicesIdx && index < kShapeIdx + shape().size()) {
    return sizeof(int64_t);
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for CSRTensor: " << ToString();
  return kTypeUnknown;
}

std::string COOTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  auto dtype = values_->Dtype();
  buf << "COOTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_->ToString() << ", values=" << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr COOTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  auto abs_sparse_tensor = std::make_shared<abstract::AbstractCOOTensor>(dtype, shape_);

  abs_sparse_tensor->set_indices(indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>());
  abs_sparse_tensor->set_values(values_->ToAbstract()->cast<abstract::AbstractTensorPtr>());

  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  abs_sparse_tensor->set_dense_shape(std::make_shared<abstract::AbstractTuple>(abstract_shape));

  return abs_sparse_tensor;
}
}  // namespace tensor
}  // namespace mindspore
