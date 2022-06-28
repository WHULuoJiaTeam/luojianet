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
@title luojianet_ms_build

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build

SET threads=8
SET ENABLE_GITEE=ON
SET ENABLE_THIRD=OFF
SET ENABLE_RS=ON

set VERSION_STR=''
for /f "tokens=1" %%a in (version.txt) do (set VERSION_STR=%%a)

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


IF "%ENABLE_THIRD%" == "ON" (
	set PATCHES_FOLDER="%BUILD_PATH%/luojianet_ms/_ms_patch"
	echo %PATCHES_FOLDER%

	IF EXIST "%PATCHES_FOLDER%" (
		rd %PATCHES_FOLDER% /S /Q
	)

	set DEPS_FOLDER="%BUILD_PATH%/luojianet_ms/_deps"
	IF EXIST "%DEPS_FOLDER%" (
		FOR /D %%p IN ("%DEPS_FOLDER%\*-src") DO rmdir "%%p" /S /Q
		FOR /D %%p IN ("%DEPS_FOLDER%\*-build") DO rmdir "%%p" /S /Q
		FOR /D %%p IN ("%DEPS_FOLDER%\*-subbuild") do (
			rmdir "%%p\CMakeFiles" /S /Q
			del "%%p\CMake*" /S /Q
			del "%%p\Make*" /S /Q
			del "%%p\cmake*" /S /Q
			echo "subpath: %%p\*-populate-prefix\src"
			FOR /D %%m IN ("%%p\*-populate-prefix") DO (
				FOR /D %%n IN ("%%m\src\*-stamp") DO rmdir "%%n" /S /Q
			)
		)
		FOR /D %%q IN ("%BUILD_PATH%\luojianet_ms") do (
			echo "newpath: %%q"
			del %%q\CMake* /S /Q
			del %%q\cmake* /S /Q
		)
		REM FOR /D %%p IN ("%BUILD_PATH%\luojianet_ms\CMake*") DO rmdir "%%p" /S /Q
	)
)

cd %BUILD_PATH%/luojianet_ms
IF "%1%" == "lite" (
    echo "======Start building LuoJiaNET Lite %VERSION_STR%======"
    rd /s /q "%BASE_PATH%\output"
    (git log -1 | findstr "^commit") > %BUILD_PATH%\.commit_id
    IF defined VisualStudioVersion (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off -DVERSION_STR=%VERSION_STR% ^
            -DCMAKE_BUILD_TYPE=Release -G "Ninja" "%BASE_PATH%/luojianet_ms/lite"
    ) ELSE (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off -DVERSION_STR=%VERSION_STR% ^
            -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" "%BASE_PATH%/luojianet_ms/lite"
    )
) ELSE (
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% -DENABLE_RS=%ENABLE_RS%^
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
