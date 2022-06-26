/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <string>
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/data_schema.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::dataset::CacheClient;
using mindspore::dataset::TaskGroup;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

// Helper function to get the session id from SESSION_ID env variable
Status GetSessionFromEnv(session_id_type *session_id) {
  RETURN_UNEXPECTED_IF_NULL(session_id);
  if (const char *session_env = std::getenv("SESSION_ID")) {
    std::string session_id_str(session_env);
    try {
      *session_id = std::stoul(session_id_str);
    } catch (const std::exception &e) {
      std::string err_msg = "Invalid numeric value for session id in env var: " + session_id_str;
      return Status(StatusCode::kMDSyntaxError, err_msg);
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Test case requires a session id to be provided via SESSION_ID environment variable.");
  }
  return Status::OK();
}

class MindDataTestCacheOp : public UT::DatasetOpTesting {
 public:
  void SetUp() override {
    DatasetOpTesting::SetUp();
    GlobalInit();
  }
};

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheServer) {
  Status rc;
  CacheClient::Builder builder;
  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  // use arbitrary session of 1, size of 0, spilling// is true
  builder.SetSessionId(env_session).SetCacheMemSz(0).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = builder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());
  // cksum value of 1 for CreateCache here...normally you do not directly create a cache and the cksum arg is generated.
  rc = myClient->CreateCache(1, true);
  ASSERT_TRUE(rc.IsOk());
  std::cout << *myClient << std::endl;

  // Create a schema using the C api's
  int32_t rank = 0;  // not used
  std::unique_ptr<DataSchema> test_schema = std::make_unique<DataSchema>();
  // 2 columns. First column is an "image" 640,480,3
  TensorShape c1Shape({640, 480, 3});
  ColDescriptor c1("image", DataType(DataType::DE_INT8), TensorImpl::kFlexible,
                   rank,  // not used
                   &c1Shape);
  // Column 2 will just be a scalar label number
  TensorShape c2Shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor c2("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, rank, &c2Shape);

  test_schema->AddColumn(c1);
  test_schema->AddColumn(c2);

  std::unordered_map<std::string, int32_t> map;
  rc = test_schema->GetColumnNameMap(&map);
  ASSERT_TRUE(rc.IsOk());

  // Test the CacheSchema api
  rc = myClient->CacheSchema(map);
  ASSERT_TRUE(rc.IsOk());

  // Create a tensor, take a snapshot and restore it back, and compare.
  std::shared_ptr<Tensor> t;
  Tensor::CreateEmpty(TensorShape({2, 3}), DataType(DataType::DE_UINT64), &t);
  t->SetItemAt<uint64_t>({0, 0}, 1);
  t->SetItemAt<uint64_t>({0, 1}, 2);
  t->SetItemAt<uint64_t>({0, 2}, 3);
  t->SetItemAt<uint64_t>({1, 0}, 4);
  t->SetItemAt<uint64_t>({1, 1}, 5);
  t->SetItemAt<uint64_t>({1, 2}, 6);
  std::cout << *t << std::endl;
  TensorTable tbl;
  TensorRow row;
  row.push_back(t);
  int64_t row_id;
  rc = myClient->WriteRow(row, &row_id);
  ASSERT_TRUE(rc.IsOk());

  // Switch off build phase.
  rc = myClient->BuildPhaseDone();
  ASSERT_TRUE(rc.IsOk());

  // Now restore from cache.
  row.clear();
  rc = myClient->GetRows({row_id}, &tbl);
  row = tbl.front();
  ASSERT_TRUE(rc.IsOk());
  auto r = row.front();
  std::cout << *r << std::endl;
  // Compare
  bool cmp = (*t == *r);
  ASSERT_TRUE(cmp);

  // Get back the schema and verify
  std::unordered_map<std::string, int32_t> map_out;
  rc = myClient->FetchSchema(&map_out);
  ASSERT_TRUE(rc.IsOk());
  cmp = (map_out == map);
  ASSERT_TRUE(cmp);

  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestCacheOp, DISABLED_TestConcurrencyRequest) {
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  TaskGroup vg;
  Status rc;

  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  // use arbitrary session of 1, size 1, spilling is true
  CacheClient::Builder builder;
  // use arbitrary session of 1, size of 0, spilling// is true
  builder.SetSessionId(env_session).SetCacheMemSz(1).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = builder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());
  // cksum value of 1 for CreateCache here...normally you do not directly create a cache and the cksum arg is generated.
  rc = myClient->CreateCache(1, true);
  ASSERT_TRUE(rc.IsOk());
  std::cout << *myClient << std::endl;
  std::shared_ptr<Tensor> t;
  Tensor::CreateEmpty(TensorShape({2, 3}), DataType(DataType::DE_UINT64), &t);
  t->SetItemAt<uint64_t>({0, 0}, 1);
  t->SetItemAt<uint64_t>({0, 1}, 2);
  t->SetItemAt<uint64_t>({0, 2}, 3);
  t->SetItemAt<uint64_t>({1, 0}, 4);
  t->SetItemAt<uint64_t>({1, 1}, 5);
  t->SetItemAt<uint64_t>({1, 2}, 6);
  TensorTable tbl;
  TensorRow row;
  row.push_back(t);
  // Cache tensor row t 5000 times using 10 threads.
  for (auto k = 0; k < 10; ++k) {
    Status vg_rc = vg.CreateAsyncTask("Test agent", [&myClient, &row]() -> Status {
      TaskManager::FindMe()->Post();
      for (auto i = 0; i < 500; i++) {
        RETURN_IF_NOT_OK(myClient->WriteRow(row));
      }
      return Status::OK();
    });
    ASSERT_TRUE(vg_rc.IsOk());
  }
  ASSERT_TRUE(vg.join_all().IsOk());
  ASSERT_TRUE(vg.GetTaskErrorIfAny().IsOk());
  rc = myClient->BuildPhaseDone();
  ASSERT_TRUE(rc.IsOk());
  // Get statistics from the server.
  CacheServiceStat stat{};
  rc = myClient->GetStat(&stat);
  ASSERT_TRUE(rc.IsOk());
  std::cout << stat.min_row_id << ":" << stat.max_row_id << ":" << stat.num_mem_cached << ":" << stat.num_disk_cached
            << "\n";
  // Expect there are 5000 rows there.
  EXPECT_EQ(5000, stat.max_row_id - stat.min_row_id + 1);
  // Get them all back using row id and compare with tensor t.
  for (auto i = stat.min_row_id; i <= stat.max_row_id; ++i) {
    tbl.clear();
    row.clear();
    rc = myClient->GetRows({i}, &tbl);
    ASSERT_TRUE(rc.IsOk());
    row = tbl.front();
    auto r = row.front();
    bool cmp = (*t == *r);
    ASSERT_TRUE(cmp);
  }
  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestCacheOp, DISABLED_TestImageFolderCacheMerge) {
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  Status rc;
  int64_t num_samples = 0;
  int64_t start_index = 0;

  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  auto seq_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  CacheClient::Builder ccbuilder;
  ccbuilder.SetSessionId(env_session).SetCacheMemSz(0).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = ccbuilder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());

  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  int32_t op_connector_size = config_manager->op_connector_size();
  std::shared_ptr<CacheLookupOp> myLookupOp =
    std::make_shared<CacheLookupOp>(4, op_connector_size, myClient, std::move(seq_sampler));
  ASSERT_NE(myLookupOp, nullptr);
  std::shared_ptr<CacheMergeOp> myMergeOp = std::make_shared<CacheMergeOp>(4, op_connector_size, 4, myClient);
  ASSERT_NE(myMergeOp, nullptr);

  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  rc = schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1));
  ASSERT_TRUE(rc.IsOk());
  rc = schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &scalar));
  ASSERT_TRUE(rc.IsOk());
  std::string dataset_path = datasets_root_path_ + "/testPK/data";
  std::set<std::string> ext = {".jpg", ".JPEG"};
  bool recursive = true;
  bool decode = false;
  std::map<std::string, int32_t> columns_to_load = {};
  std::shared_ptr<ImageFolderOp> so = std::make_shared<ImageFolderOp>(
    3, dataset_path, 3, recursive, decode, ext, columns_to_load, std::move(schema), nullptr);
  so->SetSampler(myLookupOp);
  ASSERT_TRUE(rc.IsOk());

  // RepeatOp
  uint32_t num_repeats = 4;
  std::shared_ptr<RepeatOp> myRepeatOp = std::make_shared<RepeatOp>(num_repeats);

  auto myTree = std::make_shared<ExecutionTree>();
  rc = myTree->AssociateNode(so);
  ASSERT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myLookupOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myMergeOp);
  ASSERT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssignRoot(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());

  myMergeOp->SetTotalRepeats(num_repeats);
  myMergeOp->SetNumRepeatsPerEpoch(num_repeats);
  rc = myRepeatOp->AddChild(myMergeOp);
  ASSERT_TRUE(rc.IsOk());
  myLookupOp->SetTotalRepeats(num_repeats);
  myLookupOp->SetNumRepeatsPerEpoch(num_repeats);
  rc = myMergeOp->AddChild(myLookupOp);
  ASSERT_TRUE(rc.IsOk());
  so->SetTotalRepeats(num_repeats);
  so->SetNumRepeatsPerEpoch(num_repeats);
  rc = myMergeOp->AddChild(so);
  ASSERT_TRUE(rc.IsOk());

  rc = myTree->Prepare();
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->Launch();
  ASSERT_TRUE(rc.IsOk());
  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  ASSERT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    rc = dI.FetchNextTensorRow(&tensorList);
    ASSERT_TRUE(rc.IsOk());
    if (rc.IsError()) {
      std::cout << rc << std::endl;
      break;
    }
    rowCount++;
  }
  ASSERT_EQ(rowCount, 176);
  std::cout << "Row count : " << rowCount << std::endl;
  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}
