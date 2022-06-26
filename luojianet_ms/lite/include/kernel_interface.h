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
#ifndef LUOJIANET_MS_LITE_INCLUDE_KERNEL_INTERFACE_H_
#define LUOJIANET_MS_LITE_INCLUDE_KERNEL_INTERFACE_H_

#include <memory>
#include <vector>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/lite_utils.h"
#include "schema/model_generated.h"

namespace luojianet_ms {
namespace kernel {
class Kernel;
/// \brief KernelInterface defined customized op's interface, such as infershape, and so on.
class MS_API KernelInterface {
 public:
  /// \brief Destructor of KernelInterface.
  virtual ~KernelInterface() = default;

  /// \brief Method to infer customized op's output shape.
  ///
  /// \param[in] inputs Define the input tensors of op.
  /// \param[in] outputs Define the output tensors of op.
  /// \param[in] primitive Define the attributes of op.
  ///
  /// \return  Status as a status identification of inferring.
  virtual Status Infer(std::vector<luojianet_ms::MSTensor> *inputs, std::vector<luojianet_ms::MSTensor> *outputs,
                       const schema::Primitive *primitive) {
    return kSuccess;
  }

  /// \brief Method to infer customized op's output shape.
  ///
  /// \param[in] inputs Define the input tensors of op.
  /// \param[in] outputs Define the output tensors of op.
  /// \param[in] primitive Define the attributes of op.
  /// \param[in] kernel Define the kernel of a certain op.
  ///
  /// \return  Status as a status identification of inferring.
  virtual Status Infer(std::vector<luojianet_ms::MSTensor> *inputs, std::vector<luojianet_ms::MSTensor> *outputs,
                       const schema::Primitive *primitive, const Kernel *kernel) {
    return Infer(inputs, outputs, primitive);
  }
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_INCLUDE_KERNEL_INTERFACE_H_
