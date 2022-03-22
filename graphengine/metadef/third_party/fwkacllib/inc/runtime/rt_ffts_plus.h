/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: ffts plus interface
 */

#ifndef __CCE_RUNTIME_FFTS_PLUS_H
#define __CCE_RUNTIME_FFTS_PLUS_H

#include "base.h"
#include "rt_stars_define.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

typedef struct tagFftsPlusTaskInfo {
    const rtFftsPlusSqe_t *fftsPlusSqe;
    const void *descBuf;
    size_t descBufLen;
} rtFftsPlusTaskInfo_t;

#pragma pack(pop)

RTS_API rtError_t rtGetAddrAndPrefCntWithHandle(void *handle, const void *devFunc, void **addr, uint32_t *prefetchCnt);
RTS_API rtError_t rtFftsPlusTaskLaunch(rtFftsPlusTaskInfo_t *fftsPlusTaskInfo, rtStream_t stream);

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif
