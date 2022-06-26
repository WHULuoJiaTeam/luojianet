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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_

#include <nccl.h>
#include <map>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/hal/device/distribution/collective_init.h"
#include "plugin/device/gpu/hal/hardware/nvidia_collective_comm_lib.h"
#include "runtime/collective/collective_comm_lib_loader.h"

namespace luojianet_ms {
namespace kernel {
using NvidiaCollectiveCommLib = device::gpu::NvidiaCollectiveCommLib;
static std::map<std::string, ncclDataType_t> kNcclDtypeMap = {
  {"kNumberTypeFloat32", ncclFloat}, {"kNumberTypeFloat16", ncclHalf}, {"kNumberTypeInt32", ncclInt}};

typedef ncclResult_t (*AllReduce)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, cudaStream_t,
                                  const std::string &);
typedef ncclResult_t (*AllGather)(const void *, void *, size_t, ncclDataType_t, cudaStream_t, const std::string &);
typedef ncclResult_t (*ReduceScatter)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, cudaStream_t,
                                      const std::string &);
typedef ncclResult_t (*Broadcast)(const void *, void *, size_t, ncclDataType_t, int, cudaStream_t, const std::string &);
typedef ncclResult_t (*Send)(const void *, size_t, ncclDataType_t, int, cudaStream_t, const std::string &);
typedef ncclResult_t (*Recv)(void *, size_t, ncclDataType_t, int, cudaStream_t, const std::string &);
typedef ncclResult_t (*GroupStart)();
typedef ncclResult_t (*GroupEnd)();
typedef std::vector<int> (*GetGroupRanks)(const std::string &);

class NcclGpuKernelMod : public NativeGpuKernelMod {
 public:
  NcclGpuKernelMod() : collective_handle_(nullptr), group_name_(""), nccl_data_type_(ncclHalf), use_mpi_(true) {}
  ~NcclGpuKernelMod() override = default;

 protected:
  ncclDataType_t nccl_dtype(const TypeId &type_id) { return kNcclDtypeMap[TypeIdLabel(type_id)]; }

  // Select the collective communication handle according to whether this is launched by OpenMPI or not.
  void SelectCollectiveHandle();

  // Load nvidia communication library when using LuoJiaNET communication library.
  bool LoadNvidiaCommLib();

  // The capsulation of the collective communication operation APIs for compatibility.
  // Caller does not need to judge the return value because exception will be thrown inside these methods with kernel
  // info.
  bool AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                 ncclRedOp_t reduce_op, cudaStream_t stream, const std::string &group_name);
  bool AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type, cudaStream_t stream,
                 const std::string &group_name);
  bool ReduceScatter(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                     ncclRedOp_t reduce_op, cudaStream_t stream, const std::string &group_name);
  bool Broadcast(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type, int root,
                 cudaStream_t stream, const std::string &group_name);
  bool Send(const void *send_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
            const std::string &group_name);
  bool Recv(void *recv_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
            const std::string &group_name);
  bool GroupStart();
  bool GroupEnd();

  const void *collective_handle_;
  void *nvidia_collective_handle_;
  std::string group_name_;
  ncclDataType_t nccl_data_type_;
  bool use_mpi_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_
