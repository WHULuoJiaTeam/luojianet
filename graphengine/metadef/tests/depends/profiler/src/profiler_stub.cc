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

#include "toolchain/prof_engine.h"
#include "toolchain/prof_mgr_core.h"

void * ProfMgrStartUp(const ProfMgrCfg *cfg)
{
  return nullptr;
}

int ProfMgrStop(void *handle)
{
	return 0;
}

int Msprof::Engine::RegisterEngine(const std::string& module, const Msprof::Engine::EngineIntf* engine)
{
  return 0;
}

