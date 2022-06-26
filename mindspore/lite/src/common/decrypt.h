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

#ifndef MINDSPORE_CORE_UTILS_DECRYPT_H_
#define MINDSPORE_CORE_UTILS_DECRYPT_H_

#include <string>
#include <memory>

typedef unsigned char Byte;
namespace mindspore::lite {
std::unique_ptr<Byte[]> Decrypt(const std::string &lib_path, size_t *decrypt_len, const Byte *model_data,
                                const size_t data_size, const Byte *key, const size_t key_len,
                                const std::string &dec_mode);
}  // namespace mindspore::lite
#endif
