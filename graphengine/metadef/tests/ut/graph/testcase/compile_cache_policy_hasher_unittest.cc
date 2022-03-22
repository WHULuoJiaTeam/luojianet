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
#include "graph/compile_cache_policy/compile_cache_hasher.h"
namespace ge {
class UtestCompileCachePolicyHasher : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCompileCachePolicyHasher, GetCacheDescHashWithoutShape) {
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
  auto seed = CompileCacheHasher::GetCacheDescHashWithoutShape(ccd);
  ASSERT_EQ(seed, 266203561920);

  delete data;
  data = nullptr;
}

TEST_F(UtestCompileCachePolicyHasher, GetCacheDescShapeHash) {
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
  auto seed = CompileCacheHasher::GetCacheDescShapeHash(ccd);
  ASSERT_EQ(seed, 134049801424);

  delete data;
  data = nullptr;
}
}