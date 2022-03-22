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
#include <gtest/gtest.h>
#include "graph/compile_cache_policy/compile_cache_policy.h"
#include "graph/compile_cache_policy/compile_cache_state.h"
namespace ge {
class UtestCompileCachePolicy : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCompileCachePolicy, CreateCCPSuccess_1) {
  auto ccp = ge::CompileCachePolicy::Create(MATCH_POLICY_EXACT_ONLY, AGING_POLICY_LRU);
  ASSERT_NE(ccp, nullptr);
}


TEST_F(UtestCompileCachePolicy, CreateCCPFailed_1) {
  auto mp_ptr = PolicyManager::GetInstance().GetMatchPolicy(MATCH_POLICY_EXACT_ONLY);
  auto ap_ptr = nullptr;
  auto ccp = ge::CompileCachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_EQ(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPFailed_2) {
  auto mp_ptr = nullptr;
  auto ap_ptr = nullptr;
  auto ccp = ge::CompileCachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_EQ(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, AddCacheSuccess_1) {
  int64_t uid = 100UL;
  SmallVector<ShapeType, kDefaultMaxInputNum> shapes = {{2,3,4}};
  SmallVector<ShapeType, kDefaultMaxInputNum> origin_shapes = {{2,3,4}};
  SmallVector<ShapeRangeType, kDefaultMaxInputNum> shape_ranges = {};
  SmallVector<Format, kDefaultMaxInputNum> formats = {FORMAT_ND};
  SmallVector<Format, kDefaultMaxInputNum> origin_formats = {FORMAT_ND};
  SmallVector<DataType, kDefaultMaxInputNum> data_types = {DT_FLOAT};
  uint8_t *data = new uint8_t(9);
  BinaryHolder holder = BinaryHolder();
  holder.SharedFrom(data, 1);
  SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc = {holder};
  auto ccd = CompileCacheDesc(uid, shapes, origin_shapes, shape_ranges,
                              formats, origin_formats, data_types, other_desc);

  auto ccp = ge::CompileCachePolicy::Create(MATCH_POLICY_EXACT_ONLY, AGING_POLICY_LRU);
  CacheItem cache_item = ccp->AddCache(ccd);
  ASSERT_NE(cache_item, -1);
}

TEST_F(UtestCompileCachePolicy, FindCacheSuccess_1) {
  int64_t uid = 100UL;
  SmallVector<ShapeType, kDefaultMaxInputNum> shapes = {{2,3,4}};
  SmallVector<ShapeType, kDefaultMaxInputNum> origin_shapes = {{2,3,4}};
  SmallVector<ShapeRangeType, kDefaultMaxInputNum> shape_ranges = {};
  SmallVector<Format, kDefaultMaxInputNum> formats = {FORMAT_ND};
  SmallVector<Format, kDefaultMaxInputNum> origin_formats = {FORMAT_ND};
  SmallVector<DataType, kDefaultMaxInputNum> data_types = {DT_FLOAT};
  uint8_t *data = new uint8_t(9);
  BinaryHolder holder = BinaryHolder();
  holder.SharedFrom(data, 1);
  SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc = {holder};
  auto ccd = CompileCacheDesc(uid, shapes, origin_shapes, shape_ranges,
                              formats, origin_formats, data_types, other_desc);
  auto ccp = ge::CompileCachePolicy::Create(MATCH_POLICY_EXACT_ONLY, AGING_POLICY_LRU);
  CacheItem cache_item = ccp->AddCache(ccd);

  CacheItem cache_item_find = ccp->FindCache(ccd);
  ASSERT_EQ(cache_item, cache_item_find);
}

TEST_F(UtestCompileCachePolicy, DeleteCacheSuccess_1) {
  int64_t uid = 100UL;
  SmallVector<ShapeType, kDefaultMaxInputNum> shapes = {{2,3,4}};
  SmallVector<ShapeType, kDefaultMaxInputNum> origin_shapes = {{2,3,4}};
  SmallVector<ShapeRangeType, kDefaultMaxInputNum> shape_ranges = {};
  SmallVector<Format, kDefaultMaxInputNum> formats = {FORMAT_ND};
  SmallVector<Format, kDefaultMaxInputNum> origin_formats = {FORMAT_ND};
  SmallVector<DataType, kDefaultMaxInputNum> data_types = {DT_FLOAT};
  uint8_t *data = new uint8_t(9);
  BinaryHolder holder = BinaryHolder();
  holder.SharedFrom(data, 1);
  SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc = {holder};
  auto ccd = CompileCacheDesc(uid, shapes, origin_shapes, shape_ranges,
                              formats, origin_formats, data_types, other_desc);

  auto ccp = ge::CompileCachePolicy::Create(MATCH_POLICY_EXACT_ONLY, AGING_POLICY_LRU);
  CacheItem cache_item = ccp->AddCache(ccd);

  auto lamb = [&](CacheInfo info) -> bool {
    if (info.GetItem() > 0) {
      return true;
    } else {
      return false;
    }
  };
  std::vector<CacheItem> del_item = ccp->DeleteCache(lamb);
  ASSERT_EQ(cache_item, del_item[0]);
}

TEST_F(UtestCompileCachePolicy, AgingCacheSuccess_1) {
  int64_t uid = 100UL;
  SmallVector<ShapeType, kDefaultMaxInputNum> shapes = {{2,3,4}};
  SmallVector<ShapeType, kDefaultMaxInputNum> origin_shapes = {{2,3,4}};
  SmallVector<ShapeRangeType, kDefaultMaxInputNum> shape_ranges = {};
  SmallVector<Format, kDefaultMaxInputNum> formats = {FORMAT_ND};
  SmallVector<Format, kDefaultMaxInputNum> origin_formats = {FORMAT_ND};
  SmallVector<DataType, kDefaultMaxInputNum> data_types = {DT_FLOAT};
  uint8_t *data = new uint8_t(9);
  BinaryHolder holder = BinaryHolder();
  holder.SharedFrom(data, 1);
  SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc = {holder};
  auto ccd = CompileCacheDesc(uid, shapes, origin_shapes, shape_ranges,
                              formats, origin_formats, data_types, other_desc);

  auto ccp = ge::CompileCachePolicy::Create(MATCH_POLICY_EXACT_ONLY, AGING_POLICY_LRU);
  CacheItem cache_item = ccp->AddCache(ccd);
  
  std::vector<CacheItem> del_item = ccp->DoAging();
  ASSERT_EQ(cache_item, del_item[0]);
}

}