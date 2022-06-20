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
@title luojianet_build

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build

SET threads=8
SET ENABLE_GITEE=ON
SET ENABLE_RS=ON
SET ENABLE_CACHE=ON
set ENABLE_GPU=OFF
set ENABLE_MPI=OFF
set MS_BUILD_GRPC=OFF

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

IF NOT EXIST "%BUILD_PATH%" (
    md "build"
)
cd %BUILD_PATH%
IF NOT EXIST "%BUILD_PATH%/luojianet_ms" (
    md "luojianet_ms"
)

cd %BUILD_PATH%/luojianet_ms
IF "%1%" == "lite" (
    echo "======Start building LuoJiaNet Lite %VERSION_MAJOR%.%VERSION_MINOR%.%VERSION_REVISION%======"
    rd /s /q "%BASE_PATH%\output"
    (git log -1 | findstr "^commit") > %BUILD_PATH%\.commit_id
    IF defined VisualStudioVersion (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off ^
			-DENABLE_GPU=off ^
            -DMS_VERSION_MAJOR=%VERSION_MAJOR% -DMS_VERSION_MINOR=%VERSION_MINOR% -DMS_VERSION_REVISION=%VERSION_REVISION% ^
            -DCMAKE_BUILD_TYPE=Release -G "Ninja" "%BASE_PATH%/luojianet_ms/lite"
    ) ELSE (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off ^
			-DENABLE_GPU=off ^
            -DMS_VERSION_MAJOR=%VERSION_MAJOR% -DMS_VERSION_MINOR=%VERSION_MINOR% -DMS_VERSION_REVISION=%VERSION_REVISION% ^
            -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" "%BASE_PATH%/luojianet_ms/lite"
    )
) ELSE (
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% ^
		  -DENABLE_GPU=off ^
          -G "CodeBlocks - MinGW Makefiles" ../..
)
IF NOT %errorlevel% == 0 (
    echo "cmake fail."
    call :clean
    EXIT /b 1
)

cmake --build . --target package -- -j%threads%
IF NOT %errorlevel% == 0 (
    echo "build fail."
    call :clean
    EXIT /b 1
)

call :clean
EXIT /b 0

:clean
    IF EXIST "%BASE_PATH%/output" (
        cd %BASE_PATH%/output
        if EXIST "%BASE_PATH%/output/_CPack_Packages" (
             rd /s /q _CPack_Packages
        )
    )
    cd %BASE_PATH%