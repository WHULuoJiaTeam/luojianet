/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SUBGRAPH_OPENCL_KERNEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SUBGRAPH_OPENCL_KERNEL_H_

#include <memory>
#include <set>
#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "src/runtime/gpu/opencl/opencl_allocator.h"
#include "src/runtime/gpu/opencl/opencl_executor.h"
#include "src/sub_graph_kernel.h"

namespace mindspore::kernel {
class OpenCLSubGraph : public SubGraphKernel {
 public:
  OpenCLSubGraph(const std::vector<kernel::LiteKernel *> &inKernels,
                 const std::vector<kernel::LiteKernel *> &outKernels, const std::vector<kernel::LiteKernel *> &nodes,
                 Kernel *kernel)
      : SubGraphKernel(inKernels, outKernels, nodes, kernel) {
    ocl_runtime_ = ocl_runtime_wrap_.GetInstance();
    if (nodes.front()->desc().data_type == kNumberTypeFloat16) {
      subgraph_type_ = kGpuFp16SubGraph;
      desc_.data_type = kNumberTypeFloat16;
    } else {
      subgraph_type_ = kGpuFp32SubGraph;
      desc_.data_type = kNumberTypeFloat32;
    }
    desc_.arch = kernel::KERNEL_ARCH::kGPU;
    static std::atomic_int index = 0;
    this->set_name("GpuSubGraph" + std::to_string(index++));
    nodes_set_.insert(nodes.begin(), nodes.end());
    all_kernels_infer_done_ = std::all_of(nodes_.begin(), nodes_.end(), [](const kernel::LiteKernel *kernel) {
      return kernel && kernel->InferShapeDone();
    });
  }
  ~OpenCLSubGraph() override;

  int RunPass();
  int Prepare() override;
  int ReSize() override;
  int ReSize(bool interrupt);
  int Execute() override;
  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;

 private:
  void UnInit();
  int UpdateTensorDataTypePass();
  void ReplaceOutTensorAndKernelToConvert(const lite::Tensor *in_tensor,
                                          const std::vector<kernel::LiteKernel *> &in_kernels, lite::Tensor *new_tensor,
                                          kernel::LiteKernel *in_convert_op, lite::opencl::MemType mem_type);
  void GetInOutNodes();
  int GenToFormatOp(const std::vector<lite::Tensor *> &in_tensors,
                    const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                    std::vector<lite::Tensor *> *out_tensors, std::vector<OpenCLToFormatParameter *> *out_parameters,
                    std::vector<LiteKernel *> *out_convert_ops, lite::opencl::MemType mem_type);
#ifdef ENABLE_OPENGL_TEXTURE
  int GenGLToCLOp(const std::vector<lite::Tensor *> &in_tensors,
                  const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                  std::vector<lite::Tensor *> *out_tensors,
                  std::vector<OpenGLTexture2DToOpenCLParameter *> *out_parameters,
                  std::vector<LiteKernel *> *out_convert_ops, lite::opencl::MemType mem_type);
#endif
  void GetKernelFromToTensor(const std::vector<lite::Tensor *> &in_tensors,
                             const std::vector<kernel::LiteKernel *> &in_kernels,
                             std::vector<std::vector<kernel::LiteKernel *>> *out_kernels, bool is_from);
  int FusionPass();

  int InsertOpsPass();

 public:
  using PassFunc = int (OpenCLSubGraph::*)(void);

 private:
  std::shared_ptr<lite::opencl::OpenCLAllocator> allocator_{nullptr};
  std::vector<lite::Tensor *> in_convert_tensors_;
  std::vector<lite::Tensor *> out_convert_tensors_;
  std::vector<OpenCLToFormatParameter *> in_parameters_;
  std::vector<OpenCLToFormatParameter *> out_parameters_;
#ifdef ENABLE_OPENGL_TEXTURE
  std::vector<OpenGLTexture2DToOpenCLParameter *> gl_in_parameters_;
  std::vector<OpenGLTexture2DToOpenCLParameter *> gl_out_parameters_;
#endif
  std::vector<LiteKernel *> in_convert_ops_;
  std::vector<LiteKernel *> out_convert_ops_;
  std::set<LiteKernel *> nodes_set_;
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap_;
  lite::opencl::OpenCLRuntime *ocl_runtime_{nullptr};
  bool all_kernels_infer_done_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SUBGRAPH_OPENCL_KERNEL_H_
