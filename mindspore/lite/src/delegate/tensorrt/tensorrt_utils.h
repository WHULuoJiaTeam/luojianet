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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_UTILS_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_UTILS_H_
#include <vector>
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <memory>
#include <string>
#include "src/delegate/tensorrt/op/tensorrt_op.h"
#include "src/delegate/tensorrt/cuda_impl/cublas_utils.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "schema/ops_generated.h"
#include "nnacl/pack.h"

#define kNCHW_N 0
#define kNCHW_C 1
#define kNCHW_H 2
#define kNCHW_W 3
#define kNHWC_N 0
#define kNHWC_H 1
#define kNHWC_W 2
#define kNHWC_C 3

namespace mindspore::lite {
#define TRT_VERSION_GE(major, minor) \
  (NV_TENSORRT_MAJOR > major) || ((NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR >= minor))
struct ITensorHelper;
struct ActivationParams {
  nvinfer1::ActivationType activation_type;
  bool has_alpha;
  float alpha;
  bool has_beta;
  float beta;
};

typedef union float32_bits {
  unsigned int u;
  float f;
} float32_bits;

// Convert Tensor data to Cuda dims.
nvinfer1::Dims ConvertCudaDims(const void *data, int64_t size);

nvinfer1::Dims ConvertCudaDims(int data, size_t size);

bool SameDims(nvinfer1::Dims dims, const std::vector<int64_t> &shape);

std::vector<int64_t> ConvertMSShape(const nvinfer1::Dims dims);

std::vector<int64_t> NHWC2NCHW(std::vector<int64_t> nhwc_shape);

nvinfer1::DataType ConvertDataType(DataType type_id);

cudaDataType ConvertDataType(nvinfer1::DataType type_id);

nvinfer1::IShuffleLayer *NHWC2NCHW(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &input);

nvinfer1::IShuffleLayer *NCHW2NHWC(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &input);

ActivationParams ConvertActivationType(schema::ActivationType activation_type);

nvinfer1::ITensor *ConvertConstantTensor(nvinfer1::INetworkDefinition *network, const mindspore::MSTensor &ms_tensor,
                                         const std::string &op_name);

nvinfer1::ITensor *ConvertTensorWithExpandDims(nvinfer1::INetworkDefinition *network,
                                               const mindspore::MSTensor &ms_tensor, size_t expand_shape_size,
                                               const std::string &op_name);

nvinfer1::ITensor *ConvertScalarToITensor(nvinfer1::INetworkDefinition *network, size_t shape_size, const void *value,
                                          const DataType data_type, const std::string &op_name);

nvinfer1::Weights TransposeWeight4D(const mindspore::MSTensor &ms_tensor, void **pack_weight);

nvinfer1::Weights TransposeWeight2D(const mindspore::MSTensor &ms_tensor, void **pack_weight);

nvinfer1::Weights ConvertWeight(const mindspore::MSTensor &ms_tensor);

int SetCudaDevice(std::shared_ptr<GPUDeviceInfo> device_info_);

int SetCudaDevice(int device_id);

Format GetOutputFormat(Format input_format, nvinfer1::Permutation perm);

int ConvertAxisFromNHWC2NCHW(int nhwc_axis);

void PackNHWCToNCHWFp16(const void *src, void *dst, size_t batch, size_t plane, size_t channel, size_t task_id,
                        size_t thread_count);

std::string GetTensorFormat(nvinfer1::ITensor *trt_tensor, mindspore::Format format, bool is_same);

std::string GetTensorFormat(ITensorHelper tensor_helper);

std::string GetTensorFormat(nvinfer1::ITensor *trt_tensors);

nvinfer1::ReduceOperation ConvertTRTReduceMode(schema::ReduceMode mode);

int PreprocessInputs2SameDim(nvinfer1::INetworkDefinition *network, const ITensorHelper &input_tensor_helper,
                             ITensorHelper *out_tensor_helper);

int GetDimsVolume(const nvinfer1::Dims &dims);

int GetDimsVolume(const std::vector<int64_t> &shape);

void SerializeValue(void **buffer, const void *value, size_t cpy_size);

void DeserializeValue(void const **buffer, size_t *buffer_size, void *value, size_t cpy_size);

nvinfer1::ITensor *Reshape(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input,
                           const std::vector<int64_t> &shape);

nvinfer1::ITensor *Reshape(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input,
                           const nvinfer1::Dims &shape);

template <typename T1, typename T2>
bool SameDims(const std::vector<T1> &shape1, const std::vector<T2> &shape2) {
  if (shape1.size() != shape2.size()) {
    return false;
  }
  for (size_t i = 0; i < shape1.size(); i++) {
    if (std::abs(shape1[i] - shape2[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}

template <typename T>
nvinfer1::Dims ConvertCudaDims(const std::vector<T> &shape) {
  nvinfer1::Dims dims{};
  dims.nbDims = -1;
  if (!shape.empty() && shape.size() <= static_cast<size_t>(dims.MAX_DIMS)) {
    dims.nbDims = shape.size();
    for (int i = 0; i < dims.nbDims; i++) {
      dims.d[i] = static_cast<int>(shape[i]);
    }
  } else {
    MS_LOG(WARNING) << "ms shape is invalid or empty.";
  }
  return dims;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_UTILS_H_
