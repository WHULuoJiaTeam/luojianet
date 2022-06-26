/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "common/common.h"

#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: SBUDataset.
/// Description: test basic usage of SBUDataset.
/// Expectation: get correct number of data.
TEST_F(MindDataTestPipeline, TestSBUDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSBUDataset.";

  // Create a SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds = SBU(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("caption"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SBUDatasetWithPipeline.
/// Description: test usage of SBUDataset with pipeline.
/// Expectation: get correct number of data.
TEST_F(MindDataTestPipeline, TestSBUDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSBUDatasetWithPipeline.";

  // Create two SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds1 = SBU(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  std::shared_ptr<Dataset> ds2 = SBU(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 1;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image", "caption"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("caption"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SBUIteratorOneColumn.
/// Description: test iterator of SBUDataset with only the "image" column.
/// Expectation: get correct data.
TEST_F(MindDataTestPipeline, TestSBUIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSBUIteratorOneColumn.";
  // Create a SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds = SBU(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "image" column and drop others
  std::vector<std::string> columns = {"image"};
  std::shared_ptr<Iterator> iter = ds->CreateIterator(columns, -1);
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SBUIteratorWrongColumn.
/// Description: test iterator of SBUtDataset with wrong column.
/// Expectation: get none piece of data.
TEST_F(MindDataTestPipeline, TestSBUIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSBUIteratorWrongColumn.";
  // Create a SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds = SBU(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<Iterator> iter = ds->CreateIterator(columns);
  EXPECT_EQ(iter, nullptr);
}

/// Feature: SBUDatasetSize.
/// Description: test usage of get the size of SBUDataset.
/// Expectation: get correct number of data.
TEST_F(MindDataTestPipeline, TestGetSBUDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetSBUDatasetSize.";

  // Create a SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds = SBU(folder_path, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 5);
}

/// Feature: SBUDatasetGetters.
/// Description: test usage of getters SBUDataset.
/// Expectation: get correct number of data and correct tensor shape.
TEST_F(MindDataTestPipeline, TestSBUDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSBUDatasetGetters.";

  // Create a SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds = SBU(folder_path, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 5);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "caption"};
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "string");

  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 5);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), -1);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 5);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 5);
}

/// Feature: SBUDataFail.
/// Description: test failure of SBUDataset.
/// Expectation: get none piece of data.
TEST_F(MindDataTestPipeline, TestSBUDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSBUDatasetFail.";

  // Create a SBU Dataset
  std::shared_ptr<Dataset> ds = SBU("", true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SBU input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: SBUDataWithNullSamplerFail.
/// Description: test failure of SBUDataset with null sampler.
/// Expectation: get none piece of data.
TEST_F(MindDataTestPipeline, TestSBUDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSBUDatasetWithNullSamplerFail.";

  // Create a SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds = SBU(folder_path, true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SBU input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
