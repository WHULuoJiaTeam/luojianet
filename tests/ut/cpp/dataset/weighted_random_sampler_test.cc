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
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "utils/log_adapter.h"

#include <vector>
#include <unordered_set>

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestWeightedRandomSampler : public UT::Common {
 public:
  class DummyRandomAccessOp : public RandomAccessOp {
   public:
    DummyRandomAccessOp(uint64_t num_rows) {
      // row count is in base class as protected member
      // GetNumRowsInDataset does not need an override, the default from base class is fine.
      num_rows_ = num_rows;
    }
  };
};

TEST_F(MindDataTestWeightedRandomSampler, TestOneshotReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 1000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = true
  WeightedRandomSamplerRT m_sampler(weights, num_samples, true);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }

  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}

TEST_F(MindDataTestWeightedRandomSampler, TestOneshotNoReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 1000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = replacement
  WeightedRandomSamplerRT m_sampler(weights, num_samples, false);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  // Without replacement, each sample only drawn once.
  for (int i = 0; i < total_samples; i++) {
    if (freq[i]) {
      ASSERT_EQ(freq[i], 1);
    }
  }

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}

TEST_F(MindDataTestWeightedRandomSampler, TestGetNextSampleReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 1000;
  uint64_t samples_per_tensor = 10;
  std::vector<double> weights(total_samples, std::rand() % 100);

  // create sampler with replacement = replacement
  WeightedRandomSamplerRT m_sampler(weights, num_samples, true, samples_per_tensor);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  int epoch = 0;
  while (!row.eoe()) {
    epoch++;

    for (const auto &t : row) {
      for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
        out.push_back(*it);
      }
    }

    ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  }

  ASSERT_EQ(epoch, (num_samples + samples_per_tensor - 1) / samples_per_tensor);
  ASSERT_EQ(num_samples, out.size());
}

TEST_F(MindDataTestWeightedRandomSampler, TestGetNextSampleNoReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 100;
  uint64_t samples_per_tensor = 10;
  std::vector<double> weights(total_samples, std::rand() % 100);
  weights[1] = 0;
  weights[2] = 0;
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = replacement
  WeightedRandomSamplerRT m_sampler(weights, num_samples, false, samples_per_tensor);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  int epoch = 0;
  while (!row.eoe()) {
    epoch++;

    for (const auto &t : row) {
      for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
        out.push_back(*it);
        freq[*it]++;
      }
    }

    ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  }

  // Without replacement, each sample only drawn once.
  for (int i = 0; i < total_samples; i++) {
    if (freq[i]) {
      ASSERT_EQ(freq[i], 1);
    }
  }

  ASSERT_EQ(epoch, (num_samples + samples_per_tensor - 1) / samples_per_tensor);
  ASSERT_EQ(num_samples, out.size());
}

TEST_F(MindDataTestWeightedRandomSampler, TestResetReplacement) {
  // num samples to draw.
  uint64_t num_samples = 1000000;
  uint64_t total_samples = 1000000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = true
  WeightedRandomSamplerRT m_sampler(weights, num_samples, true);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);

  m_sampler.ResetSampler();
  out.clear();

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}

TEST_F(MindDataTestWeightedRandomSampler, TestResetNoReplacement) {
  // num samples to draw.
  uint64_t num_samples = 1000000;
  uint64_t total_samples = 1000000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = true
  WeightedRandomSamplerRT m_sampler(weights, num_samples, false);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);

  m_sampler.ResetSampler();
  out.clear();
  freq.clear();
  freq.resize(total_samples, 0);
  MS_LOG(INFO) << "Resetting sampler";

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  // Without replacement, each sample only drawn once.
  for (int i = 0; i < total_samples; i++) {
    if (freq[i]) {
      ASSERT_EQ(freq[i], 1);
    }
  }

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}
