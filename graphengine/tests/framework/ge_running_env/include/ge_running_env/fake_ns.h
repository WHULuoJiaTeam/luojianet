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
#ifndef H7AEFF0EA_9FDE_487F_8562_2917A2D48EA2
#define H7AEFF0EA_9FDE_487F_8562_2917A2D48EA2

#define FAKE_NS ge
#define FAKE_NS_BEGIN namespace FAKE_NS {
#define FAKE_NS_END }
#define USING_STUB_NS using namespace FAKE_NS;
#define FWD_DECL_STUB(type) \
  namespace FAKE_NS {       \
  struct type;              \
  }

#endif
