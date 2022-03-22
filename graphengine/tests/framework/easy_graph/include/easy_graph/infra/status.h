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

#ifndef HA25033D6_1564_4748_B2C8_4DE2C5A286DE
#define HA25033D6_1564_4748_B2C8_4DE2C5A286DE

#include <stdbool.h>
#include <stdint.h>
#include "easy_graph/eg.h"

EG_NS_BEGIN

typedef uint32_t Status;

#define EG_SUCC_STATUS(status) (EG_NS::Status) status
#define EG_FAIL_STATUS(status) (EG_NS::Status)(status | EG_RESERVED_FAIL)

/* OK */
#define EG_SUCCESS EG_SUCC_STATUS(0)

/* Error Status */
#define EG_RESERVED_FAIL (EG_NS::Status) 0x80000000
#define EG_FAILURE EG_FAIL_STATUS(1)
#define EG_FATAL_BUG EG_FAIL_STATUS(2)
#define EG_TIMEDOUT EG_FAIL_STATUS(3)
#define EG_OUT_OF_RANGE EG_FAIL_STATUS(4)
#define EG_UNIMPLEMENTED EG_FAIL_STATUS(5)

static inline bool eg_status_is_ok(Status status) {
  return (status & EG_RESERVED_FAIL) == 0;
}

static inline bool eg_status_is_fail(Status status) {
  return !eg_status_is_ok(status);
}

#define __EG_FAILED(result) eg_status_is_fail(result)
#define __EG_OK(result) eg_status_is_ok(result)

EG_NS_END

#endif
