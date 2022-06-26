/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <memory>
#include <vector>
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/mind_record_sampler.h"
#include "minddata/mindrecord/include/shard_category.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "utils/log_adapter.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestMindRecordOp : public UT::DatasetOpTesting {};

// Helper function to create a MindRecordOp, sample MindRecord constructor
// MindRecordOp::MindRecordOp(int32_t num_mind_record_workers,
//                            std::vector<std::string> dataset_file, bool load_dataset, int32_t op_connector_queue_size,
//                            const std::vector<std::string> &columns_to_load,
//                            const std::vector<std::shared_ptr<ShardOperator>> &operators, int64_t num_padded,
//                            const mindrecord::json &sample_json, const std::map<std::string, std::string>
//                            &sample_bytes)

std::shared_ptr<MindRecordOp> CreateMindRecord(int32_t mind_record_workers, bool load,
                                               std::vector<std::string> dataset_files,
                                               const std::vector<std::string> &columns_to_load,
                                               const std::vector<std::shared_ptr<ShardOperator>> &operators) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  auto op_connector_queue_size = cfg->op_connector_size();
  std::map<std::string, std::string> sample_bytes = {};
  std::unique_ptr<ShardReader> shard_reader = std::make_unique<ShardReader>();
  std::shared_ptr<MindRecordSamplerRT> sampler = std::make_shared<MindRecordSamplerRT>(shard_reader.get());
  ShuffleMode shuffle_mode = ShuffleMode::kGlobal;
  std::shared_ptr<MindRecordOp> op = std::make_shared<MindRecordOp>(
    mind_record_workers, dataset_files, load, op_connector_queue_size, columns_to_load, std::move(operators), 0,
    nullptr, sample_bytes, shuffle_mode, std::move(shard_reader), std::move(sampler));
  (void)op->Init();
  return std::move(op);
}

TEST_F(MindDataTestMindRecordOp, TestMindRecordShuffle) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordShuffle";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::vector<std::shared_ptr<mindspore::mindrecord::ShardOperator>> operators;
  operators.push_back(std::make_shared<mindspore::mindrecord::ShardShuffle>(1));

  std::shared_ptr<MindRecordOp> my_mindrecord_op =
    CreateMindRecord(4, true, {mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"},
                     column_list, operators);

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  my_tree->AssociateNode(my_mindrecord_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_mindrecord_op);

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}

TEST_F(MindDataTestMindRecordOp, TestMindRecordCategory) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordCategory";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::vector<std::shared_ptr<mindspore::mindrecord::ShardOperator>> operators;
  std::vector<std::pair<std::string, std::string>> categories;
  categories.push_back(std::make_pair("label", "490"));
  categories.push_back(std::make_pair("label", "171"));
  operators.push_back(std::make_shared<mindspore::mindrecord::ShardCategory>(categories));

  std::shared_ptr<MindRecordOp> my_mindrecord_op =
    CreateMindRecord(4, true, {mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"},
                     column_list, operators);

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  my_tree->AssociateNode(my_mindrecord_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_mindrecord_op);

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}
