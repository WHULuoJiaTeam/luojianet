#!/bin/bash
# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set -e

CLANG_FORMAT=$(which clang-format) || (echo "Please install 'clang-format' tool first"; exit 1)

version=$("${CLANG_FORMAT}" --version | sed -n "s/.*\ \([0-9]*\)\.[0-9]*\.[0-9]*.*/\1/p")
if [[ "${version}" -lt "8" ]]; then
  echo "clang-format's version must be at least 8.0.0"
  exit 1
fi

CURRENT_PATH=$(pwd)
PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../}

echo "CURRENT_PATH=$CURRENT_PATH"
echo "PROJECT_HOME=$PROJECT_HOME"

# print usage message
function usage()
{
  echo "Check whether the specified source files were well formated"
  echo "Usage:"
  echo "bash $0 [-a] [-c] [-l] [-h]"
  echo "e.g. $0 -a"
  echo ""
  echo "Options:"
  echo "    -a Check code format of all files, default case"
  echo "    -c Check code format of the files changed compared to last commit"
  echo "    -l Check code format of the files changed in last commit"
  echo "    -h Print usage"
}

# check and set options
function checkopts()
{
  # init variable
  mode="all"    # default check all files

  # Process the options
  while getopts 'aclh' opt
  do
    case "${opt}" in
      a)
        mode="all"
        ;;
      c)
        mode="changed"
        ;;
      l)
        mode="lastcommit"
        ;;
      h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

# init variable
# check options
checkopts "$@"

# switch to project root path, which contains clang-format config file '.clang-format'
pushd "${CURRENT_PATH}"
    CHECK_LIST_FILE='__checked_files_list__'
    cd "${PROJECT_HOME}" || exit 1

    if [ "X${mode}" == "Xall" ]; then
      find src -type f -name "*" | grep "\.h$\|\.cc$" > "${CHECK_LIST_FILE}" || true
      find inc -type f -name "*" | grep "\.h$\|\.cc$" >> "${CHECK_LIST_FILE}" || true
    elif [ "X${mode}" == "Xchanged" ]; then
      # --diff-filter=ACMRTUXB will ignore deleted files in commit
      git diff --diff-filter=ACMRTUXB --name-only | grep "^inc\|^src" | grep "\.h$\|\.cc$" > "${CHECK_LIST_FILE}" || true
    else  # "X${mode}" == "Xlastcommit"
      git diff --diff-filter=ACMRTUXB --name-only HEAD~ HEAD | grep "^inc\|^src" | grep "\.h$\|\.cc$" > "${CHECK_LIST_FILE}" || true
    fi

    CHECK_RESULT_FILE=__code_format_check_result__
    echo "0" > "$CHECK_RESULT_FILE"

    # check format of files modified in the lastest commit 
    while read line; do
      BASE_NAME=$(basename "${line}")
      TEMP_FILE="__TEMP__${BASE_NAME}"
      cp "${line}" "${TEMP_FILE}"
      ${CLANG_FORMAT} -i "${TEMP_FILE}"
      set +e
      diff "${TEMP_FILE}" "${line}"
      ret=$?
      set -e
      rm "${TEMP_FILE}"
      if [[ "${ret}" -ne 0 ]]; then
        echo "File ${line} is not formated, please format it."
        echo "1" > "${CHECK_RESULT_FILE}"
        break
      fi
    done < "${CHECK_LIST_FILE}"

    result=$(cat "${CHECK_RESULT_FILE}")
    rm "${CHECK_RESULT_FILE}"
    rm "${CHECK_LIST_FILE}"
popd

if [[ "X${result}" == "X0" ]]; then
  echo "Check PASS: specified files are well formated!"
fi
exit "${result}"
