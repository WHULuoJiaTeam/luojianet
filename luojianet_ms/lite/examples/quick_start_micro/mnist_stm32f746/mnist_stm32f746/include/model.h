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
#ifndef LUOJIANET_MS_LITE_INCLUDE_MODEL_H_
#define LUOJIANET_MS_LITE_INCLUDE_MODEL_H_

#include "include/lite_utils.h"

namespace luojianet_ms::lite {
struct MS_API Model {
  struct Node {
    String name_;
    int node_type_;
    const void *primitive_;
    Uint32Vector input_indices_;
    Uint32Vector output_indices_;
    int quant_type_;
  };
  using NodePtrVector = Vector<Node *>;
  struct SubGraph {
    String name_;
    Uint32Vector input_indices_;
    Uint32Vector output_indices_;
    Uint32Vector node_indices_;
    Uint32Vector tensor_indices_;
  };
  using SubGraphPtrVector = Vector<SubGraph *>;
  String name_;
  String version_;
  TensorPtrVector all_tensors_;
  NodePtrVector all_nodes_;
  char *buf;
  SubGraphPtrVector sub_graphs_;

  /// \brief Static method to create a Model pointer.
  ///
  /// \param[in] model_buf Define the buffer read from a model file.
  /// \param[in] size Define bytes number of model buffer.
  ///
  /// \return Pointer of LUOJIANET_MS Lite Model.
  static Model *Import(const char *model_buf, size_t size);

  /// \brief Free meta graph temporary buffer
  virtual void Free() = 0;

  /// \brief Free all temporary buffer.EG: nodes in the model.
  virtual void Destroy() = 0;

  /// \brief Model destruct, free all memory
  virtual ~Model() = default;
};
}  // namespace luojianet_ms::lite

#endif  // LUOJIANET_MS_LITE_INCLUDE_MODEL_H_
