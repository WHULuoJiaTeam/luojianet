@rem Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
@rem Copyright 2021, 2022 Huawei Technologies Co., Ltd
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem ============================================================================
@echo off
@title gdal_build

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build

SET threads=8
SET ENABLE_GITEE=ON
SET ENABLE_RS=ON
set ENABLE_GPU=OFF
set ENABLE_MPI=OFF

set VERSION_MAJOR=''
set VERSION_MINOR=''
set ERSION_REVISION=''

for /f "delims=\= tokens=2" %%a in ('findstr /C:"const int ms_version_major = " luojianet_ms\lite\include\version.h') do (set x=%%a)
set VERSION_MAJOR=%x:~1,1%
for /f "delims=\= tokens=2" %%b in ('findstr /C:"const int ms_version_minor = " luojianet_ms\lite\include\version.h') do (set y=%%b)
set VERSION_MINOR=%y:~1,1%
for /f "delims=\= tokens=2" %%c in ('findstr /C:"const int ms_version_revision = " luojianet_ms\lite\include\version.h') do (set z=%%c)
set VERSION_REVISION=%z:~1,1%

ECHO %2%|FINDSTR "^[0-9][0-9]*$"
IF %errorlevel% == 0 (
    SET threads=%2%
)

IF "%FROM_GITEE%" == "1" (
    echo "DownLoad from gitee"
    SET ENABLE_GITEE=ON
)


cd %BASE_PATH%/third_party
IF "%ENABLE_RS%" == "ON" (
	call gdal.bat
)