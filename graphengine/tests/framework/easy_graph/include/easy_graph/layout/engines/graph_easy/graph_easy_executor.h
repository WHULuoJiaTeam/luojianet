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

#ifndef HAC96EB3A_2169_4BB0_A8EB_7B966C262B2F
#define HAC96EB3A_2169_4BB0_A8EB_7B966C262B2F

#include "easy_graph/layout/layout_executor.h"

EG_NS_BEGIN

struct GraphEasyExecutor : LayoutExecutor {
 private:
  Status Layout(const Graph &, const LayoutOption *) override;
};

EG_NS_END

#endif
