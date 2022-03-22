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

#include <gtest/gtest.h>

#include "common/auth/file_saver.h"

namespace ge {
class UTEST_file_saver : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
 
TEST_F(UTEST_file_saver, save_model_data_to_buff_success) {
  ModelFileHeader file_header;
  std::vector<char> data;
  data.resize(sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo), 0);
  ModelPartitionTable *partition_table = reinterpret_cast<ModelPartitionTable*>(data.data());
  partition_table->num = 1;
  partition_table->partition[0] = { MODEL_DEF, 0, 12 };
  std::vector<ModelPartitionTable *> partition_tables;
  partition_tables.push_back(partition_table);
  auto buff = reinterpret_cast<uint8_t *>(malloc(12));
  struct ge::ModelPartition model_partition;
  model_partition.type = MODEL_DEF;
  model_partition.data = buff;
  model_partition.size = 12;
  std::vector<ModelPartition> model_partitions = { model_partition };
  std::vector<std::vector<ModelPartition>> all_partition_datas = { model_partitions };
  ge::ModelBufferData model;

  Status ret = FileSaver::SaveToBuffWithFileHeader(file_header, partition_tables, all_partition_datas, model);
  EXPECT_EQ(ret, ge::SUCCESS);

  free(buff);
  buff = nullptr;
  model_partition.data = nullptr;
}
}  // namespace ge