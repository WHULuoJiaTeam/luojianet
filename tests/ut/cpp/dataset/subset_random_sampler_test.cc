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
#include "gtest/gtest.h"

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"

#include <vector>
#include <unordered_set>

using namespace mindspore::dataset;

class MindDataTestSubsetRandomSampler : public UT::Common {
 public:
  class DummyRandomAccessOp : public RandomAccessOp {
   public:
    DummyRandomAccessOp(int64_t num_rows) {
      num_rows_ = num_rows;  // base class
    };
  };
};

TEST_F(MindDataTestSubsetRandomSampler, TestAllAtOnce) {
  std::vector<int64_t> in({0, 1, 2, 3, 4});
  std::unordered_set<int64_t> in_set(in.begin(), in.end());
  int64_t num_samples = 0;
  SubsetRandomSamplerRT sampler(in, num_samples);

  DummyRandomAccessOp dummyRandomAccessOp(5);
  sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<int64_t> out;
  ASSERT_EQ(sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<int64_t>(); it != t->end<int64_t>(); it++) {
      out.push_back(*it);
    }
  }
  ASSERT_EQ(in.size(), out.size());
  for (int i = 0; i < in.size(); i++) {
    ASSERT_NE(in_set.find(out[i]), in_set.end());
  }

  ASSERT_EQ(sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}

TEST_F(MindDataTestSubsetRandomSampler, TestGetNextSample) {
  int64_t total_samples = 100000 - 5;
  int64_t samples_per_tensor = 10;
  int64_t num_samples = 0;
  std::vector<int64_t> input(total_samples, 1);
  SubsetRandomSamplerRT sampler(input, num_samples, samples_per_tensor);

  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<int64_t> out;

  ASSERT_EQ(sampler.GetNextSample(&row), Status::OK());
  int epoch = 0;
  while (!row.eoe()) {
    epoch++;
    for (const auto &t : row) {
      for (auto it = t->begin<int64_t>(); it != t->end<int64_t>(); it++) {
        out.push_back(*it);
      }
    }

    ASSERT_EQ(sampler.GetNextSample(&row), Status::OK());
  }

  ASSERT_EQ(epoch, (total_samples + samples_per_tensor - 1) / samples_per_tensor);
  ASSERT_EQ(input.size(), out.size());
}

TEST_F(MindDataTestSubsetRandomSampler, TestReset) {
  std::vector<int64_t> in({0, 1, 2, 3, 4});
  std::unordered_set<int64_t> in_set(in.begin(), in.end());
  int64_t num_samples = 0;
  SubsetRandomSamplerRT sampler(in, num_samples);

  DummyRandomAccessOp dummyRandomAccessOp(5);
  sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<int64_t> out;

  ASSERT_EQ(sampler.GetNextSample(&row), Status::OK());
  for (const auto &t : row) {
    for (auto it = t->begin<int64_t>(); it != t->end<int64_t>(); it++) {
      out.push_back(*it);
    }
  }
  ASSERT_EQ(in.size(), out.size());
  for (int i = 0; i < in.size(); i++) {
    ASSERT_NE(in_set.find(out[i]), in_set.end());
  }

  sampler.ResetSampler();

  ASSERT_EQ(sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), false);
  out.clear();
  for (const auto &t : row) {
    for (auto it = t->begin<int64_t>(); it != t->end<int64_t>(); it++) {
      out.push_back(*it);
    }
  }
  ASSERT_EQ(in.size(), out.size());
  for (int i = 0; i < in.size(); i++) {
    ASSERT_NE(in_set.find(out[i]), in_set.end());
  }

  ASSERT_EQ(sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}
