/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: ffts plus interface
 */

#ifndef CCE_RUNTIME_RT_FFTS_PLUS_H
#define CCE_RUNTIME_RT_FFTS_PLUS_H

#include "base.h"
#include "rt_ffts_plus_define.h"
#include "rt_stars_define.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

typedef struct tagFftsPlusTaskInfo {
    const rtFftsPlusSqe_t *fftsPlusSqe;
    const void *descBuf;           // include total context
    size_t      descBufLen;        // the length of descBuf
} rtFftsPlusTaskInfo_t;

#pragma pack(pop)

RTS_API rtError_t rtGetAddrAndPrefCntWithHandle(void *hdl, const void *devFunc, void **addr, uint32_t *prefetchCnt);

RTS_API rtError_t rtFftsPlusTaskLaunch(rtFftsPlusTaskInfo_t *fftsPlusTaskInfo, rtStream_t stm);

RTS_API rtError_t rtFftsPlusTaskLaunchWithFlag(rtFftsPlusTaskInfo_t *fftsPlusTaskInfo, rtStream_t stm,
                                               uint32_t flag);

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // CCE_RUNTIME_RT_FFTS_PLUS_H