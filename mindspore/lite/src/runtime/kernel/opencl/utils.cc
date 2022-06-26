/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/utils.h"
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include "src/kernel_registry.h"
#include "src/common/file_utils.h"

using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_TANH;

namespace mindspore::kernel {
const std::set<schema::PrimitiveType> ArithmeticPrimitives = {schema::PrimitiveType_MulFusion,
                                                              schema::PrimitiveType_AddFusion,
                                                              schema::PrimitiveType_SubFusion,
                                                              schema::PrimitiveType_DivFusion,
                                                              schema::PrimitiveType_LogicalAnd,
                                                              schema::PrimitiveType_LogicalOr,
                                                              schema::PrimitiveType_Maximum,
                                                              schema::PrimitiveType_Minimum,
                                                              schema::PrimitiveType_FloorDiv,
                                                              schema::PrimitiveType_FloorMod,
                                                              schema::PrimitiveType_SquaredDifference,
                                                              schema::PrimitiveType_Equal,
                                                              schema::PrimitiveType_NotEqual,
                                                              schema::PrimitiveType_Less,
                                                              schema::PrimitiveType_LessEqual,
                                                              schema::PrimitiveType_Greater,
                                                              schema::PrimitiveType_GreaterEqual,
                                                              schema::PrimitiveType_Eltwise,
                                                              schema::PrimitiveType_BiasAdd};

const std::set<schema::PrimitiveType> ArithmeticSelfPrimitives = {
  schema::PrimitiveType_Abs,        schema::PrimitiveType_Ceil,  schema::PrimitiveType_Cos,
  schema::PrimitiveType_ExpFusion,  schema::PrimitiveType_Floor, schema::PrimitiveType_Log,
  schema::PrimitiveType_LogicalNot, schema::PrimitiveType_Round, schema::PrimitiveType_Rsqrt,
  schema::PrimitiveType_Sin,        schema::PrimitiveType_Neg,   schema::PrimitiveType_Sqrt,
  schema::PrimitiveType_Square};

std::string GetActDefines() {
  static std::string act_defines = "#define ActivationType_RELU " + std::to_string(ActivationType_RELU) +
                                   "\n#define ActivationType_RELU6 " + std::to_string(ActivationType_RELU6) +
                                   "\n#define ActivationType_LEAKY_RELU " + std::to_string(ActivationType_LEAKY_RELU) +
                                   "\n#define ActivationType_TANH " + std::to_string(ActivationType_TANH) +
                                   "\n#define ActivationType_SIGMOID " + std::to_string(ActivationType_SIGMOID) + "\n";
  return act_defines;
}

int GetUpPow2(int n) {
  int i = 0;
  int j = 0;
  while (n > 0) {
    j += n & 1;
    n = n >> 1;
    i++;
  }
  return 1 << (i - (j == 1));
}

int GetMaxDivisor(int x, int divisor) {
  int i = divisor;
  while (i > 0) {
    if (x % i == 0) {
      return i;
    }
    i--;
  }
  return 1;
}

int GetMaxDivisorStrategy0(int x, int divisor) {
  if (divisor >= C8NUM && x % C8NUM == 0) {
    return C8NUM;
  } else if (divisor >= C4NUM && x % C4NUM == 0) {
    return C4NUM;
  } else if (divisor >= C2NUM && x % C2NUM == 0) {
    return C2NUM;
  } else {
    return GetMaxDivisor(x, divisor);
  }
}

int GetMaxDivisorStrategy1(int x, int divisor) {
  if (divisor >= C8NUM && x % C8NUM == 0) {
    return x / C8NUM;
  } else if (divisor >= C4NUM && x % C4NUM == 0) {
    return x / C4NUM;
  } else if (divisor >= C2NUM && x % C2NUM == 0) {
    return x / C2NUM;
  } else {
    return GetMaxDivisor(x, divisor);
  }
}

std::map<cl_int, std::string> error_infos = {
  {CL_SUCCESS, "Success"},
  {CL_DEVICE_NOT_FOUND, "Device not found"},
  {CL_DEVICE_NOT_AVAILABLE, "Device not available"},
  {CL_COMPILER_NOT_AVAILABLE, "Compiler not available"},
  {CL_MEM_OBJECT_ALLOCATION_FAILURE, "Memory object allocation failure"},
  {CL_OUT_OF_RESOURCES, "Out of resources"},
  {CL_OUT_OF_HOST_MEMORY, "Out of host memory"},
  {CL_PROFILING_INFO_NOT_AVAILABLE, "Profiling information not available"},
  {CL_MEM_COPY_OVERLAP, "Memory copy overlap"},
  {CL_IMAGE_FORMAT_MISMATCH, "Image format mismatch"},
  {CL_IMAGE_FORMAT_NOT_SUPPORTED, "Image format not supported"},
  {CL_BUILD_PROGRAM_FAILURE, "Build program failure"},
  {CL_MAP_FAILURE, "Mapping failure"},
  {CL_MISALIGNED_SUB_BUFFER_OFFSET, "Misaligned sub-buffer offset"},
  {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "Execution status error for events in wait list"},
  {CL_COMPILE_PROGRAM_FAILURE, "Compile program failure"},
  {CL_LINKER_NOT_AVAILABLE, "Linker not available"},
  {CL_LINK_PROGRAM_FAILURE, "Link program failure"},
  {CL_DEVICE_PARTITION_FAILED, "Device partition failed"},
  {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "Kernel argument information not available"},
  {CL_INVALID_VALUE, "Invalid value"},
  {CL_INVALID_DEVICE_TYPE, "Invalid device type"},
  {CL_INVALID_PLATFORM, "Invalid platform"},
  {CL_INVALID_DEVICE, "Invalid device"},
  {CL_INVALID_CONTEXT, "Invalid context"},
  {CL_INVALID_QUEUE_PROPERTIES, "Invalid queue properties"},
  {CL_INVALID_COMMAND_QUEUE, "Invalid command queue"},
  {CL_INVALID_HOST_PTR, "Invalid host pointer"},
  {CL_INVALID_MEM_OBJECT, "Invalid memory object"},
  {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "Invalid image format descriptor"},
  {CL_INVALID_IMAGE_SIZE, "Invalid image size"},
  {CL_INVALID_SAMPLER, "Invalid sampler"},
  {CL_INVALID_BINARY, "Invalid binary"},
  {CL_INVALID_BUILD_OPTIONS, "Invalid build options"},
  {CL_INVALID_PROGRAM, "Invalid program"},
  {CL_INVALID_PROGRAM_EXECUTABLE, "Invalid program executable"},
  {CL_INVALID_KERNEL_NAME, "Invalid kernel name"},
  {CL_INVALID_KERNEL_DEFINITION, "Invalid kernel definition"},
  {CL_INVALID_KERNEL, "Invalid kernel"},
  {CL_INVALID_ARG_INDEX, "Invalid argument index"},
  {CL_INVALID_ARG_VALUE, "Invalid argument value"},
  {CL_INVALID_ARG_SIZE, "Invalid argument size"},
  {CL_INVALID_KERNEL_ARGS, "Invalid kernel arguments"},
  {CL_INVALID_WORK_DIMENSION, "Invalid work dimension"},
  {CL_INVALID_WORK_GROUP_SIZE, "Invalid work group size"},
  {CL_INVALID_WORK_ITEM_SIZE, "Invalid work item size"},
  {CL_INVALID_GLOBAL_OFFSET, "Invalid global offset"},
  {CL_INVALID_EVENT_WAIT_LIST, "Invalid event wait list"},
  {CL_INVALID_EVENT, "Invalid event"},
  {CL_INVALID_OPERATION, "Invalid operation"},
  {CL_INVALID_GL_OBJECT, "Invalid GL object"},
  {CL_INVALID_BUFFER_SIZE, "Invalid buffer size"},
  {CL_INVALID_MIP_LEVEL, "Invalid mip-level"},
  {CL_INVALID_GLOBAL_WORK_SIZE, "Invalid global work size"},
  {CL_INVALID_PROPERTY, "Invalid property"},
  {CL_INVALID_IMAGE_DESCRIPTOR, "Invalid image descriptor"},
  {CL_INVALID_COMPILER_OPTIONS, "Invalid compiler options"},
  {CL_INVALID_LINKER_OPTIONS, "Invalid linker options"},
  {CL_INVALID_DEVICE_PARTITION_COUNT, "Invalid device partition count"},
  {CL_INVALID_PIPE_SIZE, "Invalid pipe size"},
  {CL_INVALID_DEVICE_QUEUE, "Invalid device queue"},
  {CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR, "Invalid GL share group reference KHR"}};

std::string CLErrorCode(cl_int error_code) {
  auto it = error_infos.find(error_code);
  if (it == error_infos.end()) {
    return "Unknown OpenCL error code";
  } else {
    return it->second;
  }
}

int GetBroadcastGpuAxis(int ndim, int ori_axis) {
  if (ori_axis >= ndim) {
    return ndim - 1;
  }
  int axis = 0;
  if (ndim == DIMENSION_1D) {
    axis = kNHWC_C;
  } else if (ndim == DIMENSION_2D) {
    axis = ori_axis == kNHWC_N ? kNHWC_N : kNHWC_C;
  } else if (ndim == DIMENSION_3D) {
    axis = ori_axis == kNHWC_N ? kNHWC_N : ori_axis == kNHWC_H ? kNHWC_W : kNHWC_C;
  } else if (ndim == DIMENSION_4D) {
    axis = ori_axis;
  } else if (ndim > DIMENSION_4D) {
    MS_LOG(ERROR) << "GPU doesn't support ndim>=" << ndim;
  }
  return axis;
}

#ifdef ENABLE_FP16
void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                     int data_type) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  auto src_fp16 = reinterpret_cast<float16_t *>(src);
  auto src_fp32 = reinterpret_cast<float32_t *>(src);
  auto src_int32 = reinterpret_cast<int32_t *>(src);
  auto dst_fp16 = reinterpret_cast<float16_t *>(dst);
  auto dst_fp32 = reinterpret_cast<float32_t *>(dst);
  auto dst_int32 = reinterpret_cast<int32_t *>(dst);
  for (int n = 0, src_idx = 0; n < tensor.N; n++) {
    for (int h = 0; h < tensor.D * tensor.H; ++h) {
      for (int w = 0; w < tensor.W; ++w) {
        for (int c = 0; c < tensor.C; ++c, ++src_idx) {
          int dst_idx = ((n * tensor.D * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
          if (data_type == kNumberTypeInt32) {
            dst_int32[dst_idx] = src_int32[src_idx];
          } else if (dst_is_fp16) {
            dst_fp16[dst_idx] = src_is_fp16 ? src_fp16[src_idx] : static_cast<float16_t>(src_fp32[src_idx]);
          } else {
            dst_fp32[dst_idx] = src_is_fp16 ? static_cast<float32_t>(src_fp16[src_idx]) : src_fp32[src_idx];
          }
        }
      }
    }
  }
  // scalar
  if (tensor.ElementsNum == 1) {
    if (dst_is_fp16) {
      dst_fp16[kNHWC_C] = dst_fp16[kNHWC_W] = dst_fp16[kNHWC_H] = dst_fp16[kNHWC_N];
    } else {
      dst_fp32[kNHWC_C] = dst_fp32[kNHWC_W] = dst_fp32[kNHWC_H] = dst_fp32[kNHWC_N];
    }
  }
}
#else
void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                     int data_type) {
  MS_ASSERT(dst);
  MS_ASSERT(src);
  auto src_fp32 = reinterpret_cast<float *>(src);
  auto src_int32 = reinterpret_cast<int32_t *>(src);
  auto dst_fp32 = reinterpret_cast<float *>(dst);
  auto dst_int32 = reinterpret_cast<int32_t *>(dst);
  for (size_t n = 0, src_idx = 0; n < tensor.N; n++) {
    for (size_t h = 0; h < tensor.D * tensor.H; ++h) {
      for (size_t w = 0; w < tensor.W; ++w) {
        for (size_t c = 0; c < tensor.C; ++c, ++src_idx) {
          int dst_idx = ((n * tensor.D * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
          if (data_type == kNumberTypeInt32) {
            dst_int32[dst_idx] = src_int32[src_idx];
          } else {
            dst_fp32[dst_idx] = src_fp32[src_idx];
          }
        }
      }
    }
  }
  // scalar
  if (tensor.ElementsNum == 1) {
    dst_fp32[kNHWC_C] = dst_fp32[kNHWC_W] = dst_fp32[kNHWC_H] = dst_fp32[kNHWC_N];
  }
}
#endif

#ifdef ENABLE_FP16
void PackNCHWToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                     int data_type) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  auto src_int32 = reinterpret_cast<int32_t *>(src);
  auto src_fp32 = reinterpret_cast<float32_t *>(src);
  auto src_fp16 = reinterpret_cast<float16_t *>(src);
  auto dst_int32 = reinterpret_cast<int32_t *>(dst);
  auto dst_fp32 = reinterpret_cast<float32_t *>(dst);
  auto dst_fp16 = reinterpret_cast<float16_t *>(dst);
  for (int src_idx = 0, n = 0; n < tensor.N; n++) {
    for (int c = 0; c < tensor.C; ++c) {
      for (int h = 0; h < tensor.D * tensor.H; ++h) {
        for (int w = 0; w < tensor.W; ++w, ++src_idx) {
          int dst_idx = ((n * tensor.D * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
          if (data_type == kNumberTypeInt32) {
            dst_int32[dst_idx] = src_int32[src_idx];
          } else if (dst_is_fp16) {
            dst_fp16[dst_idx] = src_is_fp16 ? src_fp16[src_idx] : static_cast<float16_t>(src_fp32[src_idx]);
          } else {
            dst_fp32[dst_idx] = src_is_fp16 ? static_cast<float32_t>(src_fp16[src_idx]) : src_fp32[src_idx];
          }
        }
      }
    }
  }
  // scalar
  if (tensor.ElementsNum == 1) {
    if (dst_is_fp16) {
      dst_fp16[kNHWC_N] = dst_fp16[kNHWC_H] = dst_fp16[kNHWC_W] = dst_fp16[kNHWC_C];
    } else {
      dst_fp32[kNHWC_N] = dst_fp32[kNHWC_H] = dst_fp32[kNHWC_W] = dst_fp32[kNHWC_C];
    }
  }
}
#else
void PackNCHWToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                     int data_type) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  auto src_fp32 = reinterpret_cast<float *>(src);
  auto src_int32 = reinterpret_cast<int32_t *>(src);
  auto dst_fp32 = reinterpret_cast<float *>(dst);
  auto dst_int32 = reinterpret_cast<int32_t *>(dst);
  for (size_t n = 0, src_idx = 0; n < tensor.N; n++) {
    for (size_t c = 0; c < tensor.C; ++c) {
      for (size_t h = 0; h < tensor.D * tensor.H; ++h) {
        for (size_t w = 0; w < tensor.W; ++w, ++src_idx) {
          int dst_idx = ((n * tensor.D * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
          if (data_type == kNumberTypeInt32) {
            dst_int32[dst_idx] = src_int32[src_idx];
          } else {
            dst_fp32[dst_idx] = src_fp32[src_idx];
          }
        }
      }
    }
  }
  // scalar
  if (tensor.ElementsNum == 1) {
    dst_fp32[kNHWC_C] = dst_fp32[kNHWC_W] = dst_fp32[kNHWC_H] = dst_fp32[kNHWC_N];
  }
}
#endif

int CheckParamLikeTensor(const std::string &kernel_name, const std::string &tensor_name, lite::Tensor *tensor,
                         TypeId expect_data_type, const std::vector<int> &expect_shape) {
  if (!tensor->IsConst()) {
    MS_LOG(WARNING) << "in " << kernel_name << ": tensor " << tensor_name << " must be Const.";
    return RET_ERROR;
  }
  if (tensor->data_type() != expect_data_type) {
    MS_LOG(WARNING) << "in " << kernel_name << ": tensor's data_type must be " << expect_data_type;
    return RET_ERROR;
  }
  if (tensor->shape() != expect_shape) {
    std::string expect_shape_str = "(";
    for (auto i : expect_shape) {
      expect_shape_str += std::to_string(i) + ",";
    }
    expect_shape_str += ")";

    std::string tensor_shape_str = "(";
    for (auto i : tensor->shape()) {
      tensor_shape_str += std::to_string(i) + ",";
    }
    tensor_shape_str += ")";

    MS_LOG(WARNING) << "in " << kernel_name
                    << ": tensor's shape is error. expect_shape: " + expect_shape_str +
                         " tensor->shape(): " + tensor_shape_str;
    return RET_ERROR;
  }
  return RET_OK;
}

void *StoreTensorData(lite::Tensor *tensor) {
  if ((tensor != nullptr) && (tensor->data() != nullptr) && (tensor->Size() > 0)) {
    void *stored_data = malloc(tensor->Size());
    if (stored_data == nullptr) {
      MS_LOG(ERROR) << "StoreTensorData Malloc Failed.";
      return nullptr;
    }
    memcpy(stored_data, tensor->data(), tensor->Size());
    return stored_data;
  }
  return nullptr;
}

void FreeStoredData(void *data) {
  if (data != nullptr) {
    free(data);
  }
}

std::vector<std::string> CreateBuildOptionsExtByDType(TypeId type_id) {
  std::vector<std::string> build_options_ext;
  if (type_id == kNumberTypeInt32) {
    build_options_ext = {" -DDTYPE=int -DDTYPE4=int4 -DWRITE_IMAGE=write_imagei  -DREAD_IMAGE=read_imagei "};
  } else if (type_id == kNumberTypeFloat32) {
    build_options_ext = {" -DDTYPE=float -DDTYPE4=float4 -DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagef "};
  } else if (type_id == kNumberTypeFloat16) {
    build_options_ext = {" -DDTYPE=half -DDTYPE4=half4 -DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imageh "};
  }
  return build_options_ext;
}
}  // namespace mindspore::kernel
