/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/skip_first_epoch_sampler.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

using namespace mindspore::dataset;

Status CreateINT64Tensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements, unsigned char *data = nullptr) {
  TensorShape shape(std::vector<int64_t>(1, num_elements));
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(shape, DataType(DataType::DE_INT64), data, sample_ids));

  return Status::OK();
}

class MindDataTestStandAloneSampler : public UT::DatasetOpTesting {
 protected:
  class MockStorageOp : public RandomAccessOp {
   public:
    MockStorageOp(int64_t val) {
      // row count is in base class as protected member
      // GetNumRowsInDataset does not need an override, the default from base class is fine.
      num_rows_ = val;
    }
  };
};

TEST_F(MindDataTestStandAloneSampler, TestDistributedSampler) {
  std::vector<std::shared_ptr<Tensor>> row;
  uint64_t res[6][7] = {{0, 3, 6, 9, 12, 15, 18},  {1, 4, 7, 10, 13, 16, 19}, {2, 5, 8, 11, 14, 17, 0},
                        {0, 17, 4, 10, 14, 8, 15}, {13, 9, 16, 3, 2, 19, 12}, {1, 11, 6, 18, 7, 5, 0}};
  for (int i = 0; i < 6; i++) {
    std::shared_ptr<Tensor> t;
    Tensor::CreateFromMemory(TensorShape({7}), DataType(DataType::DE_INT64), (unsigned char *)(res[i]), &t);
    row.push_back(t);
  }
  MockStorageOp mock(20);
  std::shared_ptr<Tensor> tensor;
  int64_t num_samples = 0;
  TensorRow sample_row;
  for (int i = 0; i < 6; i++) {
    std::shared_ptr<SamplerRT> sampler =
      std::make_shared<DistributedSamplerRT>(3, i % 3, (i < 3 ? false : true), num_samples);
    sampler->HandshakeRandomAccessOp(&mock);
    sampler->GetNextSample(&sample_row);
    tensor = sample_row[0];
    MS_LOG(DEBUG) << (*tensor);
    if (i < 3) {  // This is added due to std::shuffle()
      EXPECT_TRUE((*tensor) == (*row[i]));
    }
  }
}

TEST_F(MindDataTestStandAloneSampler, TestStandAoneSequentialSampler) {
  std::vector<std::shared_ptr<Tensor>> row;
  MockStorageOp mock(5);
  uint64_t res[5] = {0, 1, 2, 3, 4};
  std::shared_ptr<Tensor> label1, label2;
  CreateINT64Tensor(&label1, 3, reinterpret_cast<unsigned char *>(res));
  CreateINT64Tensor(&label2, 2, reinterpret_cast<unsigned char *>(res + 3));
  int64_t num_samples = 0;
  int64_t start_index = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples, 3);

  std::shared_ptr<Tensor> tensor;
  TensorRow sample_row;
  sampler->HandshakeRandomAccessOp(&mock);
  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label1));
  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label2));
  sampler->ResetSampler();
  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label1));
  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label2));
}

/// Feature: MindData RT SkipFirstEpochSampler Support
/// Description: Test MindData RT SkipFirstEpochSampler by Checking Sample Outputs
/// Expectation: Results are successfully outputted.
TEST_F(MindDataTestStandAloneSampler, TestStandAloneSkipFirstEpochSampler) {
  MS_LOG(INFO) << "Doing MindDataTestStandAloneSampler-TestStandAloneSkipFirstEpochSampler.";
  std::vector<std::shared_ptr<Tensor>> row;
  MockStorageOp mock(5);
  uint64_t res[5] = {0, 1, 2, 3, 4};
  std::shared_ptr<Tensor> label, label1, label2;
  CreateINT64Tensor(&label, 5, reinterpret_cast<unsigned char *>(res));
  CreateINT64Tensor(&label1, 3, reinterpret_cast<unsigned char *>(res));
  CreateINT64Tensor(&label2, 2, reinterpret_cast<unsigned char *>(res + 3));
  int64_t num_samples = 0;
  int64_t start_index = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<SkipFirstEpochSamplerRT>(start_index, num_samples, 3);

  std::shared_ptr<Tensor> tensor;
  TensorRow sample_row;
  sampler->HandshakeRandomAccessOp(&mock);
  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label1));

  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label2));

  // Test output after Reset
  sampler->ResetSampler();
  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label));

  // Test different start index
  start_index = 2;
  CreateINT64Tensor(&label, 3, reinterpret_cast<unsigned char *>(res + 2));
  sampler = std::make_shared<SkipFirstEpochSamplerRT>(start_index, num_samples, 3);
  sampler->HandshakeRandomAccessOp(&mock);
  sampler->GetNextSample(&sample_row);
  tensor = sample_row[0];
  EXPECT_TRUE((*tensor) == (*label));
}