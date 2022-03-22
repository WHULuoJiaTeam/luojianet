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

#ifndef INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_STORE_H_
#define INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_STORE_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "./ge_task_info.h"
#include "./ops_kernel_info_types.h"
#include "cce/aicpu_engine_struct.h"
#include "cce/fwk_adpt_struct.h"
#include "common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "proto/task.pb.h"
using std::map;
using std::string;
using std::to_string;
using std::vector;

namespace ge {
class OpDesc;

class OpsKernelInfoStore {
 public:
  OpsKernelInfoStore() {}

  virtual ~OpsKernelInfoStore() {}

  // initialize opsKernelInfoStore
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  // close opsKernelInfoStore
  virtual Status Finalize() = 0; /*lint -e148*/

  virtual Status CreateSession(const std::map<std::string, std::string> &session_options) { return SUCCESS; }

  virtual Status DestroySession(const std::map<std::string, std::string> &session_options) { return SUCCESS; }

  // get all opsKernelInfo
  virtual void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const = 0;

  // whether the opsKernelInfoStore is supported based on the operator attribute
  virtual bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const = 0;

  virtual bool CheckAccuracySupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason,
                                      const bool realQuery = false) const {
    return CheckSupported(opDescPtr, un_supported_reason);
  }
  // opsFlag opsFlag[0] indicates constant folding is supported or not
  virtual void opsFlagCheck(const ge::Node &node, std::string &opsFlag) {};

  // only call fe engine interface to compile single op
  virtual Status CompileOp(std::vector<ge::NodePtr> &node_vec) { return SUCCESS; }
  virtual Status CompileOpRun(std::vector<ge::NodePtr> &node_vec) { return SUCCESS; }
  // load task for op
  virtual Status LoadTask(GETaskInfo &task) { return SUCCESS; }

  virtual bool CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason) const {
    if (node == nullptr) {
      return false;
    }
    return CheckSupported(node->GetOpDesc(), un_supported_reason);
  }

  virtual bool CheckAccuracySupported(const ge::NodePtr &node, std::string &un_supported_reason,
                                      const bool realQuery = false) const {
    if (node == nullptr) {
      return false;
    }
    return CheckAccuracySupported(node->GetOpDesc(), un_supported_reason, realQuery);
  }
  // Set cut support info
  virtual Status SetCutSupportedInfo(const ge::NodePtr &node) { return SUCCESS; }
};
}  // namespace ge
#endif  // INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_STORE_H_
