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
#ifndef LUOJIANET_MS_LITE_SRC_TRAIN_TRAIN_EXPORT_H_
#define LUOJIANET_MS_LITE_SRC_TRAIN_TRAIN_EXPORT_H_
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <utility>
#include <set>
#include "schema/inner/model_generated.h"
#include "src/lite_kernel.h"
#include "src/lite_model.h"
#include "include/train/train_cfg.h"

namespace luojianet_ms {
#ifndef _STUB
namespace schema {
struct CNodeT;
struct TensorT;
struct MetaGraphT;
}  // namespace schema
#endif
namespace lite {
struct tensor_info {
  size_t input_index;
  OpParameter *op_parameter;
};

class TrainExport {
 public:
  explicit TrainExport(const std::string file_name) : file_name_(file_name) {}
  virtual ~TrainExport();
  int ExportNet(const std::vector<luojianet_ms::kernel::LiteKernel *> &kernels,
                const std::vector<luojianet_ms::lite::Tensor *> &tensors, const std::vector<std::string> &output_names,
                const Model *model, QuantizationType quant_type);
  int ExportInit(const std::string model_name, std::string version);
  int SaveToFile();
  void set_connect(const std::unordered_map<size_t, size_t> &map) { connect_ = map; }
  int LoadModel(void *buf, size_t buf_size);
  int AddTransformNode();
  int TrainModelFusion();
  int TrainModelDrop();

 protected:
  virtual std::vector<uint8_t> CreateData(const luojianet_ms::lite::Tensor *tensor);

 private:
  std::string file_name_;
  schema::MetaGraphT *meta_graph_ = nullptr;
  std::vector<size_t> out_idx_;
  std::map<size_t, size_t> remap_;
  std::unordered_map<size_t, size_t> connect_;  // connection map (backbone tenor id-> head tensor id)
  bool IsNodeNonDepend(const std::unique_ptr<schema::CNodeT> &node, const std::vector<size_t> &sinked_tensor_idxes);
  int TopologicalSort();
  void PrepareRemap(int offset);
  Model::Node *FindNode(const luojianet_ms::kernel::LiteKernel *kernel, const Model *model);
  std::unique_ptr<schema::TensorT> CreateTensor(const Tensor *tensor, schema::Tensor *scTensor, int preferred_dim,
                                                const int tensor_quant_type);
  std::unique_ptr<schema::CNodeT> CreateCNode(const luojianet_ms::kernel::LiteKernel *kernel,
                                              std::vector<uint32_t> inputIndex, std::vector<uint32_t> outputIndex,
                                              const Model *model);
  bool IsInputTensor(const schema::TensorT &t);
  int CreateAndAddCNode(const luojianet_ms::kernel::LiteKernel *kernel, std::vector<uint32_t> inputIndex,
                        std::vector<uint32_t> outputIndex, const Model *model);
  std::unique_ptr<schema::CNodeT> CreateTransformNode(std::vector<uint32_t> inputIndex,
                                                      std::vector<uint32_t> outputIndex, size_t id);
  std::unique_ptr<schema::TensorT> CreateTransformTensor(size_t id);
  std::unique_ptr<schema::TensorT> CreateTransformConst(size_t last_id);
  int AddTransform();
  bool NeedQuantization(const luojianet_ms::lite::Tensor *tensor, const int tensor_quant_type);
  int ExportTensor(const Model *model, const std::vector<luojianet_ms::lite::Tensor *> &tensors, int offset,
                   const std::vector<std::pair<size_t, tensor_info>> &map_index,
                   const std::vector<std::string> &output_names, const std::set<size_t> &out_set);
  virtual int QuantTensorData(schema::TensorT *dest_tensor, const luojianet_ms::lite::Tensor *src_tensor,
                              int preferred_dim);
  luojianet_ms::schema::QuantType GetNodeQuantType(const luojianet_ms::kernel::LiteKernel *kernel);
  void TagQuantizedNodes();
  QuantizationType quant_type_;
};
};  // namespace lite
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_SRC_TRAIN_TRAIN_EXPORT_H_
