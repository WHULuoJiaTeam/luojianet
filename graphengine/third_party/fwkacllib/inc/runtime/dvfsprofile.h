/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: dvfsprofile.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_DVFSPROFILE_H
#define CCE_RUNTIME_DVFSPROFILE_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum dvfsProfileMode {
  DVFS_PROFILE_PERFORMANCE_PRIORITY,
  DVFS_PROFILE_BALANCE_PRIORITY,
  DVFS_PROFILE_POWER_PRIORITY,
  DVFS_PROFILE_PRIORITY_MAX
} DvfsProfileMode;

/**
 * @ingroup dvrt_dvfsprofile
 * @brief Set the performance mode of the device
 * @param [in] profMode   dvfsProfileMode
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetDvfsProfile(DvfsProfileMode profMode);

/**
 * @ingroup dvrt_dvfsprofile
 * @brief Set the performance mode of the device
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for invalid value
 */
RTS_API rtError_t rtUnsetDvfsProfile();

/**
 * @ingroup dvrt_dvfsprofile
 * @brief Get the current performance mode of the device
 * @param [in|out] pmode   dvfsProfileMode type pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetDvfsProfile(DvfsProfileMode *pmode);

#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_DVFSPROFILE_H
