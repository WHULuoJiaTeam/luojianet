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

#ifndef INC_FRAMEWORK_MEMORY_MEMORY_API_H_
#define INC_FRAMEWORK_MEMORY_MEMORY_API_H_

#include "external/ge/ge_api_error_codes.h"
#include "runtime/mem.h"

namespace ge {
enum MemStorageType {
  HBM = 0,
  RDMA_HBM,
  HOST_DDR,
};

struct HostVarInfo {
  uint64_t base_addr;
  uint64_t var_size;
};

struct TensorInfo {
  std::string var_name;
  std::vector<int64_t> dims;
  DataType data_type;
};

///
/// \param size [in] rdma pool memory size to be allocated.
/// \param mem_type [in] memory type for rdma pool.
/// \return Status result of function
GE_FUNC_VISIBILITY Status InitRdmaPool(size_t size, rtMemType_t mem_type = RT_MEMORY_HBM);

///
/// \param var_info [in] host variable addr infos.
/// \param mem_type [in] memory type for rdma pool.
/// \return Status result of function
GE_FUNC_VISIBILITY Status RdmaRemoteRegister(const std::vector<HostVarInfo> &var_info,
                                             rtMemType_t mem_type = RT_MEMORY_HBM);

///
/// \param tensor_info [in] description for tensor stored shared memory.
/// \param dev_addr [out] malloced shared memory addr.
/// \param memory_size [out] malloced shared memory size.
/// \return Status result of function
GE_FUNC_VISIBILITY Status MallocSharedMemory(const TensorInfo &tensor_info, uint64_t &dev_addr, uint64_t &memory_size);

///
/// \param var_name [in] var_name name of host variable.
/// \param base_addr [out] base_addr vase addr of host variable.
/// \param var_size [out] var_size memory_size of host variable.
/// \return Status result of function
GE_FUNC_VISIBILITY Status GetVarBaseAddrAndSize(const std::string &var_name, uint64_t &base_addr, uint64_t &var_size);
}  // namespace ge
#endif  // INC_FRAMEWORK_MEMORY_MEMORY_API_H_
