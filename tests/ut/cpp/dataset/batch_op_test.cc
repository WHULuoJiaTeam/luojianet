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
#include <memory>
#include <string>
#include "minddata/dataset/core/client.h"
// #include "minddata/dataset/core/pybind_support.h"
// #include "minddata/dataset/core/tensor.h"
// #include "minddata/dataset/core/tensor_shape.h"
// #include "minddata/dataset/engine/datasetops/batch_op.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"
#include "minddata/dataset/util/status.h"
// #include "pybind11/numpy.h"
// #include "pybind11/pybind11.h"

// #include "utils/ms_utils.h"

// #include "minddata/dataset/engine/db_connector.h"
// #include "minddata/dataset/kernels/data/data_utils.h"

namespace common = mindspore::common;
namespace de = mindspore::dataset;

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

class MindDataTestBatchOp : public UT::DatasetOpTesting {
 protected:
};

// This test has been disabled because PadInfo is not currently supported in the C++ API.
// Feature: Test Batch op with padding on TFReader
// Description: Create Batch operation with padding on a TFReader dataset
// Expectation: The data within the created object should match the expected data
TEST_F(MindDataTestBatchOp, DISABLED_TestSimpleBatchPadding) {
  std::string schema_file = datasets_root_path_ + "/testBatchDataset/test.data";
  PadInfo m;
  std::shared_ptr<Tensor> pad_value;
  Tensor::CreateEmpty(TensorShape::CreateScalar(), DataType(DataType::DE_FLOAT32), &pad_value);
  pad_value->SetItemAt<float>({}, -1);
  m.insert({"col_1d", std::make_pair(TensorShape({4}), pad_value)});

  /*
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  auto op_connector_size = config_manager->op_connector_size();
  auto num_workers = config_manager->num_parallel_workers();
  std::vector<std::string> input_columns = {};
  std::vector<std::string> output_columns = {};
  pybind11::function batch_size_func;
  pybind11::function batch_map_func;
  */

  int32_t batch_size = 12;
  bool drop = false;
  std::shared_ptr<BatchOp> op = Batch(batch_size, drop, m);
  //  std::make_shared<BatchOp>(batch_size, drop, pad, op_connector_size, num_workers, input_columns, output_columns,
  //                            batch_size_func, batch_map_func, m);
  auto tree = Build({TFReader(schema_file), op});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
  } else {
    int64_t payload[] = {-9223372036854775807 - 1,
                         1,
                         -1,
                         -1,
                         2,
                         3,
                         -1,
                         -1,
                         4,
                         5,
                         -1,
                         -1,
                         6,
                         7,
                         -1,
                         -1,
                         8,
                         9,
                         -1,
                         -1,
                         10,
                         11,
                         -1,
                         -1,
                         12,
                         13,
                         -1,
                         -1,
                         14,
                         15,
                         -1,
                         -1,
                         16,
                         17,
                         -1,
                         -1,
                         18,
                         19,
                         -1,
                         -1,
                         20,
                         21,
                         -1,
                         -1,
                         22,
                         23,
                         -1,
                         -1};
    std::shared_ptr<de::Tensor> t;
    rc = de::Tensor::CreateFromMemory(de::TensorShape({12, 4}), de::DataType(DataType::DE_INT64),
                                      (unsigned char *)payload, &t);
    de::DatasetIterator di(tree);
    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE((*t) == (*(tensor_map["col_1d"])));
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(tensor_map.size() == 0);
    EXPECT_TRUE(rc.IsOk());
  }
}
