/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TRANSFER_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TRANSFER_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {
class TransferNode : public DatasetNode {
 public:
  /// \brief Constructor
  TransferNode(std::shared_ptr<DatasetNode> child, std::string queue_name, std::string device_type, int32_t device_id,
               bool send_epoch_end, int32_t total_batch, bool create_data_info_queue);

  /// \brief Destructor
  ~TransferNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kTransferNode; }

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  static Status get_distribution(std::shared_ptr<DatasetNode> ds, int32_t *device_id);

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(IRNodePass *const p, bool *const modified) override;

  /// \brief Getter functions
  const std::string &QueueName() const { return queue_name_; }
  int32_t DeviceId() const { return device_id_; }
  const std::string &DeviceType() const { return device_type_; }
  bool SendEpochEnd() const { return send_epoch_end_; }
  int32_t TotalBatch() const { return total_batch_; }
  bool CreateDataInfoQueue() const { return create_data_info_queue_; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

  /// \brief Function for read dataset operation from json
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[in] ds dataset node constructed
  /// \param[out] result Deserialized dataset after the operation
  /// \return Status The status code returned
  static Status from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                          std::shared_ptr<DatasetNode> *result);

 private:
  std::string queue_name_;
  int32_t device_id_;
  std::string device_type_;
  bool send_epoch_end_;
  int32_t total_batch_;
  bool create_data_info_queue_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TRANSFER_NODE_H_
