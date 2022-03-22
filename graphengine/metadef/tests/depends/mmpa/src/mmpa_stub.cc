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

#include "mmpa/mmpa_api.h"

typedef int mmErrorMSg;

INT32 mmOpen(const CHAR *path_name, INT32 flags) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (NULL == path_name) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if (0 == (flags & (O_RDONLY | O_WRONLY | O_RDWR | O_CREAT))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = open(path_name, flags);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmOpen2(const CHAR *path_name, INT32 flags, MODE mode) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (NULL == path_name) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if (MMPA_ZERO == (flags & (O_RDONLY | O_WRONLY | O_RDWR | O_CREAT))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }
  if ((MMPA_ZERO == (mode & (S_IRUSR | S_IREAD))) && (MMPA_ZERO == (mode & (S_IWUSR | S_IWRITE)))) {
    syslog(LOG_ERR, "The permission mode of the file is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = open(path_name, flags, mode);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmClose(INT32 fd) {
  INT32 result = EN_OK;

  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "The file fd is invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = close(fd);
  if (EN_OK != result) {
    syslog(LOG_ERR, "Close the file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return EN_OK;
}

mmSsize_t mmWrite(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  mmSsize_t result = MMPA_ZERO;

  if ((fd < MMPA_ZERO) || (NULL == mm_buf)) {
    syslog(LOG_ERR, "Input parameter invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = write(fd, mm_buf, (size_t)mm_count);
  if (result < MMPA_ZERO) {
    syslog(LOG_ERR, "Write buf to file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return result;
}

mmSsize_t mmRead(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  mmSsize_t result = MMPA_ZERO;

  if ((fd < MMPA_ZERO) || (NULL == mm_buf)) {
    syslog(LOG_ERR, "Input parameter invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = read(fd, mm_buf, (size_t)mm_count);
  if (result < MMPA_ZERO) {
    syslog(LOG_ERR, "Read file to buf failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return result;
}

INT32 mmMkdir(const CHAR *lp_path_name, mmMode_t mode) {
  INT32 t_mode = mode;
  INT32 ret = EN_OK;

  if (NULL == lp_path_name) {
    syslog(LOG_ERR, "The input path is null.\r\n");
    return EN_INVALID_PARAM;
  }

  if (t_mode < MMPA_ZERO) {
    syslog(LOG_ERR, "The input mode is wrong.\r\n");
    return EN_INVALID_PARAM;
  }

  ret = mkdir(lp_path_name, mode);

  if (EN_OK != ret) {
    syslog(LOG_ERR, "Failed to create the directory, the ret is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return EN_OK;
}

void *memCpyS(void *dest, const void *src, UINT32 count) {
  char *tmp = (char *)dest;
  char *s = (char *)src;

  if (MMPA_ZERO == count) {
    return dest;
  }

  while (count--) {
    *tmp++ = *s++;
  }
  return dest;
}

INT32 mmRmdir(const CHAR *lp_path_name) { return rmdir(lp_path_name); }

mmTimespec mmGetTickCount() {
  mmTimespec rts;
  struct timespec ts = {0};
  (void)clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  rts.tv_sec = ts.tv_sec;
  rts.tv_nsec = ts.tv_nsec;
  return rts;
}

INT32 mmGetTid() {
  INT32 ret = (INT32)syscall(SYS_gettid);

  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }

  return ret;
}

INT32 mmAccess(const CHAR *path_name) {
  if (path_name == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = access(path_name, F_OK);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmStatGet(const CHAR *path, mmStat_t *buffer) {
  if ((path == NULL) || (buffer == NULL)) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = stat(path, buffer);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmGetFileSize(const CHAR *file_name, ULONGLONG *length) {
  if ((file_name == NULL) || (length == NULL)) {
    return EN_INVALID_PARAM;
  }
  struct stat file_stat;
  (void)memset_s(&file_stat, sizeof(file_stat), 0, sizeof(file_stat));  // unsafe_function_ignore: memset
  INT32 ret = lstat(file_name, &file_stat);
  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }
  *length = (ULONGLONG)file_stat.st_size;
  return EN_OK;
}

INT32 mmScandir(const CHAR *path, mmDirent ***entryList, mmFilter filterFunc,  mmSort sort)
{
  return 0;
}

VOID mmScandirFree(mmDirent **entryList, INT32 count)
{
}

INT32 mmAccess2(const CHAR *pathName, INT32 mode)
{
  return 0;
}

INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone)
{
  return 0;
}

INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen)
{
  INT32 ret = EN_OK;
  if ((path == NULL) || (realPath == NULL) || (realPathLen < MMPA_MAX_PATH)) {
    return EN_INVALID_PARAM;
  }
  const CHAR *ptr = realpath(path, realPath);
  if (ptr == NULL) {
    ret = EN_ERROR;
  }
  return ret;
}

//INT32 mmGetErrorCode()
//{
  //return 0;
//}

INT32 mmIsDir(const CHAR *fileName)
{
  return 0;
}

INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len)
{
  return 0;
}

INT32 mmDlclose(VOID *handle)
{
  return 0;
}

CHAR *mmDlerror()
{
  return "";
}

INT32 mmDladdr(VOID *addr, mmDlInfo *info)
{
  return 0;
}

VOID *mmDlopen(const CHAR *fileName, INT32 mode)
{
  return NULL;
}

VOID *mmDlsym(VOID *handle, const CHAR *funcName)
{
  return NULL;
}

INT32 mmGetPid()
{
  return (INT32)getpid();
}

INT32 mmGetErrorCode(VOID)
{
  INT32 ret = (INT32)errno;
  return ret;
}

CHAR *mmGetErrorFormatMessage(mmErrorMSg errnum, CHAR *buf, mmSize size)
{
  if ((buf == NULL) || (size <= 0)) {
    return NULL;
  }
  return strerror_r(errnum, buf, size);
}

