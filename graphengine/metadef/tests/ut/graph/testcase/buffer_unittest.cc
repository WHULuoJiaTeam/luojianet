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
#include <securec.h>
#include <gtest/gtest.h>
#include "graph/buffer.h"

namespace ge {
class BufferUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(BufferUT, ShareFrom1) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateShareFrom(buf); // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, ShareFrom2) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  BufferUtils::ShareFrom(buf, buf1); // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, OperatorAssign) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  buf1 = buf; // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateShareFrom) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateShareFrom(buf);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateCopyFrom1) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 2;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 2;
  }
  second_buf[50] = 250;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  BufferUtils::CopyFrom(buf, buf1);
  buf1.GetData()[50] = 250;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateCopyFrom2) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 2;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 2;
  }
  second_buf[50] = 250;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateCopyFrom(buf);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 250;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}
}  // namespace ge