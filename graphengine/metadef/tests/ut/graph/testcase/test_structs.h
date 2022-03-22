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

#ifndef METADEF_CXX_TEST_STRUCTS_H
#define METADEF_CXX_TEST_STRUCTS_H
namespace ge {
struct TestStructA {
  TestStructA(int64_t a, int64_t b, int64_t c) : a(a), b(b), c(c) {}
  int64_t a;
  int64_t b;
  int64_t c;
  bool operator==(const TestStructA &other) const {
    return a == other.a && b == other.b && c == other.c;
  }
};

struct InlineStructB {
  InlineStructB() {
    a = new int32_t[10]();
  }
  InlineStructB(const InlineStructB  &other) {
    a = new int32_t[10]();
    memcpy(a, other.a, sizeof(int32_t[10]));
  }
  InlineStructB(InlineStructB &&other) {
    a = other.a;
    other.a = nullptr;
  }
  InlineStructB &operator=(const InlineStructB &other) {
    memcpy(a, other.a, sizeof(int32_t[10]));
    return *this;
  }
  InlineStructB &operator=(InlineStructB &&other) noexcept {
    delete[] a;
    a = other.a;
    other.a = nullptr;
    return *this;
  }
  bool operator==(const InlineStructB &other) const {
    return memcmp(a, other.a, sizeof(int32_t) * 10) == 0;
  }
  ~InlineStructB() {
    delete[] a;
  }

  InlineStructB &Set(size_t index, int32_t value) {
    a[index] = value;
    return *this;
  }

  int32_t Get(size_t index) const {
    return a[index];
  }
  int32_t *GetP() {
    return a;
  }
 private:
  int32_t *a;
};
}
#endif  //METADEF_CXX_TEST_STRUCTS_H
