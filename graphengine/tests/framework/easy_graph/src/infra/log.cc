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

#include <iostream>
#include <string.h>
#include <stdarg.h>
#include "easy_graph/infra/log.h"

EG_NS_BEGIN

namespace {
struct ConsoleState {
  bool isColorful() const {
    return colorful;
  }

 protected:
  ConsoleState(bool c) : colorful(c) {}

 private:
  bool colorful;
};

////////////////////////////////////////////////////////
#define DEF_COLOR_STATE(STATE, COLOR)                                                                                  \
  struct STATE : ConsoleState {                                                                                        \
    STATE(bool colorful) : ConsoleState(colorful) {}                                                                   \
  };                                                                                                                   \
  std::ostream &operator<<(std::ostream &os, const STATE &state) {                                                     \
    if (state.isColorful())                                                                                            \
      os << COLOR;                                                                                                     \
    return os;                                                                                                         \
  }

////////////////////////////////////////////////////////
#define __RED "\033[1;31m"
#define __GREEN "\033[1;32m"
#define __YELLOW "\033[1;33m"
#define __BLUE "\033[1;34m"
#define __MAGENTA "\033[1;35m"
#define __CYAN "\033[1;36m"
#define __WHITE "\033[0m"

DEF_COLOR_STATE(DebugState, __BLUE)
DEF_COLOR_STATE(InfoState, __CYAN)
DEF_COLOR_STATE(NormalState, __WHITE)
DEF_COLOR_STATE(SuccState, __GREEN)
DEF_COLOR_STATE(WarnState, __YELLOW)
DEF_COLOR_STATE(FailState, __RED)

///////////////////////////////////////////////////////
struct StdoutListener;
StdoutListener *inst = nullptr;

struct StdoutListener {
  static StdoutListener &getInstance() {
    if (inst == 0) {
      inst = new StdoutListener(true);
    }

    return *inst;
  }

  void println(const uint8_t level, const char *msg) {
#define ON_LEVEL(level, state)                                                                                         \
  case level:                                                                                                          \
    doPrint(state, msg);                                                                                               \
    break;

    switch (level) {
      ON_LEVEL(EG_FATAL_LEVEL, fail)
      ON_LEVEL(EG_ERR_LEVEL, fail)
      ON_LEVEL(EG_WARN_LEVEL, warn)
      ON_LEVEL(EG_SUCC_LEVEL, succ)
      ON_LEVEL(EG_INFO_LEVEL, info)
      ON_LEVEL(EG_DEBUG_LEVEL, debug)
      default:
        doPrint(normal, msg);
    }
  }

 private:
  StdoutListener(bool colorful)
      : succ(colorful), fail(colorful), normal(colorful), info(colorful), warn(colorful), debug(colorful) {}

  template<typename STATE>
  void doPrint(const STATE &state, const char *msg) {
    std::cout << state << msg << normal << std::endl;
  }

 private:
  SuccState succ;
  FailState fail;
  NormalState normal;
  InfoState info;
  WarnState warn;
  DebugState debug;
};

const char *getBaseName(const char *absPath) {
  if (absPath == 0)
    return "";
  const char *p = absPath + strlen(absPath);
  while ((p != absPath) && (*(p - 1) != '/')) {
    p--;
  }
  return p;
}

}  // namespace

void eg_log(int level, const char *levelstr, const char *file, unsigned int line, const char *fmt, ...) {
  const int FMT_BUFF_SIZE = 1024;
  char fmt_buff[FMT_BUFF_SIZE] = {0};
  va_list valist;
  va_start(valist, fmt);
  vsnprintf(fmt_buff, FMT_BUFF_SIZE, fmt, valist);
  va_end(valist);

  char buff[1280] = {0};
  sprintf(buff, "[%s]: %s:%u: %s", levelstr, getBaseName(file), line, fmt_buff);
  StdoutListener::getInstance().println(level, buff);
}

EG_NS_END
