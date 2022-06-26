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

#include "coder/generator/component/const_blocks/mcontext.h"

namespace mindspore::lite::micro {
const char context_header[] = R"RAW(
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_CONTEXT_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_CONTEXT_H_

#include <stdbool.h>

typedef struct MicroContext {
  char* vendor_name_;
  int thread_num_; /**< thread number config for thread pool */
  bool enable_parallel_;
  int* affinity_core_list_; /**< explicitly specify the core to be bound. priority use affinity core list */
  int core_num;
  int affinity_mode;
} MicroContext;

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_CONTEXT_H_
)RAW";

const char context_source[] = R"RAW(
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "context.h"
#include "c_api/context_c.h"
#include <stdlib.h>
#include <string.h>

MSContextHandle MSContextCreate() {
  MicroContext *micro_context = (MicroContext *)malloc(sizeof(MicroContext));
  if (micro_context == NULL) {
    return NULL;
  }
  micro_context->enable_parallel_ = false;
  micro_context->thread_num_ = 1;
  micro_context->affinity_core_list_ = NULL;
  micro_context->core_num = 0;
  micro_context->affinity_mode = 0;
  return micro_context;
}

void MSContextDestroy(MSContextHandle *context) {
  MicroContext *micro_context = (MicroContext *)(*context);
  if (micro_context) {
    if (micro_context->affinity_core_list_) {
      free(micro_context->affinity_core_list_);
      micro_context->affinity_core_list_ = NULL;
    }
    free(micro_context);
    micro_context = NULL;
  }
}

void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    micro_context->thread_num_ = thread_num;
  }
}

int32_t MSContextGetThreadNum(const MSContextHandle context) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    return micro_context->thread_num_;
  }
  return 0;
}

void MSContextSetThreadAffinityMode(MSContextHandle context, int mode) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    micro_context->affinity_mode = mode;
  }
}

int MSContextGetThreadAffinityMode(const MSContextHandle context) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    return micro_context->affinity_mode;
  }
  return 0;
}

void MSContextSetThreadAffinityCoreList(MSContextHandle context, const int32_t *core_list, size_t core_num) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    int *micro_cores = (int *)malloc(sizeof(int32_t) * core_num);
    (void)memcpy(micro_cores, core_list, sizeof(int32_t) * core_num);
    micro_context->affinity_core_list_ = micro_cores;
  }
}

const int32_t *MSContextGetThreadAffinityCoreList(const MSContextHandle context, size_t *core_num) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    return micro_context->affinity_core_list_;
  }
  return NULL;
}

void MSContextSetEnableParallel(MSContextHandle context, bool is_parallel) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    micro_context->enable_parallel_ = is_parallel;
  }
}

bool MSContextGetEnableParallel(const MSContextHandle context) {
  MicroContext *micro_context = (MicroContext *)context;
  if (micro_context) {
    return micro_context->enable_parallel_;
  }
  return false;
}
)RAW";
}  // namespace mindspore::lite::micro
