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

#include "graph/load/model_manager/task_info/task_info.h"

#include <vector>

namespace ge {
Status TaskInfo::SetStream(uint32_t stream_id, const std::vector<rtStream_t> &stream_list) {
  if (stream_list.size() == 1) {
    stream_ = stream_list[0];
  } else if (stream_list.size() > stream_id) {
    stream_ = stream_list[stream_id];
  } else {
  	REPORT_INNER_ERROR("E19999", "stream_id:%u >= stream_list.size(): %zu, check invalid",
                      stream_id, stream_list.size());
    GELOGE(FAILED, "[Check][Param] index:%u >= stream_list.size():%zu.", stream_id, stream_list.size());
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
