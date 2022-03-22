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

#ifndef INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
#define INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_

#include <cstdint>
#include <string>
#include <vector>
#include "runtime/rt.h"

using std::string;
namespace ge {
// when need to eliminate GETaskKernelHcclInfo, so not need DAVINCI_TRAIN/DAVINCI_CLOUD
struct GETaskKernelHcclInfo {
  std::string input_name;
  std::string hccl_type;
  void *inputDataAddr;
  void *outputDataAddr;
  void *workSpaceAddr;
  int32_t count;
  int32_t dataType;
  int32_t opType;
  int64_t rootId;
  uint64_t workSpaceMemSize;
  std::vector<int64_t> dims;
  std::vector<rtStream_t> hcclStreamList;
};

struct GETaskInfo {
  uint32_t id;
  uint16_t type;
  uint32_t streamID;
  void *stream;  // rtKernelLaunch input argument
  void *event;
  void *privateDef;
  uint32_t privateDefLen;
  void *opsKernelStorePtr;

  std::vector<GETaskKernelHcclInfo> kernelHcclInfo;
};

struct HcomOpertion {
  std::string hcclType;
  void *inputPtr;
  void *outputPtr;
  uint64_t count;
  int32_t dataType;
  int32_t opType;
  int32_t root;
};

struct HcomRemoteAccessAddrInfo
{
  uint32_t remotetRankID;
  uint64_t remoteAddr;  // host embedding table address
  uint64_t localAddr;  // device HBM address
  uint64_t length;   // memory Length in Bytes
};


}  // namespace ge
#endif  // INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
