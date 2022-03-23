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

SET GDAL_FOLDER="lib_gdal"
if not exist %GDAL_FOLDER% (
	git clone https://gitee.com/mizhangwhuer/lib_gdal.git
)

cd lib_gdal

SET BUILD_FOLDER="build"
if not exist %BUILD_FOLDER% (
    mkdir build
)
cd build
cmake .. -DWITH_ZLIB=ON -DWITH_ZLIB_EXTERNAL=ON -DWITH_EXPAT=ON -DWITH_EXPAT_EXTERNAL=ON -DWITH_JSONC=ON -DWITH_JSONC_EXTERNAL=ON -DWITH_ICONV=ON -DWITH_CURL=ON -DWITH_CURL_EXTERNAL=ON -DWITH_LibXml2=ON -DWITH_LibXml2_EXTERNAL=ON -DWITH_GEOS=ON -DWITH_GEOS_EXTERNAL=ON -DWITH_JPEG=ON -DWITH_JPEG_EXTERNAL=ON -DWITH_JPEG12=OFF -DWITH_JPEG12_EXTERNAL=OFF -DWITH_TIFF=ON -DWITH_TIFF_EXTERNAL=ON -DWITH_GeoTIFF=ON -DWITH_GeoTIFF_EXTERNAL=ON -DWITH_JBIG=ON -DWITH_JBIG_EXTERNAL=ON -DWITH_GIF=ON -DWITH_GIF_EXTERNAL=ON -DWITH_OpenCAD=ON -DWITH_OpenCAD_EXTERNAL=ON -DWITH_PNG=ON -DWITH_PNG_EXTERNAL=ON -DWITH_PROJ=ON -DWITH_PROJ_EXTERNAL=ON -DWITH_OpenJPEG=ON -DWITH_OpenJPEG_EXTERNAL=ON -DENABLE_OPENJPEG=ON -DWITH_OpenSSL=OFF -DWITH_OpenSSL_EXTERNAL=OFF -DWITH_LibLZMA=ON -DWITH_LibLZMA_EXTERNAL=ON -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DENABLE_OZI=ON -DENABLE_NITF_RPFTOC_ECRGTOC=ON -DGDAL_ENABLE_GNM=ON -DWITH_OCI=ON -DWITH_OCI_EXTERNAL=ON -DENABLE_OCI=ON -DENABLE_GEORASTER=ON -DWITH_SQLite3=ON -DWITH_SQLite3_EXTERNAL=ON -DWITH_PostgreSQL=OFF -DWITH_PostgreSQL_EXTERNAL=OFF -WITH_Boost=ON -DWITH_Boost_EXTERNAL=ON -DWITH_KML=OFF -DWITH_KML_EXTERNAL=OFF -DGDAL_BUILD_APPS=ON -DWITH_HDF4=ON -DWITH_HDF4_EXTERNAL=ON -DENABLE_HDF4=ON -DWITH_QHULL=OFF -DWITH_QHULL_EXTERNAL=OFF -DWITH_Spatialite=OFF -DWITH_Spatialite_EXTERNAL=OFF -DWITH_SZIP=ON -DWITH_SZIP_EXTERNAL=ON -DWITH_UriParser=ON -DWITH_UriParser_EXTERNAL=ON -DWITH_NUMPY=ON -DENABLE_WEBP=ON -DWITH_WEBP=OFF -DWITH_WEBP_EXTERNAL=OFF -DBUILD_TESTING=ON -DSKIP_PYTHON_TESTS=ON -DWITH_GTest=ON -DWITH_GTest_EXTERNAL=ON -DWITH_NUMPY=OFF -DWITH_NUMPY_EXTERNAL=OFF -DWITH_ICONV=ON -DWITH_ICONV_EXTERNAL=ON  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles"
cmake --build . --config Release --target package

SET SRC_DIR="%cd%/_CPack_Packages/win64/ZIP/gdal-3.2.1-GNU-7.3"
SET DST_DIR="%cd%/../../GDAL_win"
SET DLL_SRC_DIR="%cd%/third-party/install/bin"
SET DLL_DST_DIR="%cd%/../../GDAL_win/lib/site-packages"
SET GDAL_DLL_SRC_DIR="%cd%/_CPack_Packages/win64/ZIP/gdal-3.2.1-GNU-7.3/bin"
SET GDAL_DLL_DST_DIR="%cd%\..\..\GDAL_win\lib\site-packages"
SET extension=dll

SET GIF_DLL_FILE="%cd%\..\..\GDAL_win\lib\site-packages\libgif.dll"
SET REN_GIF_DLL_FILE=gif.dll

SET PYD_SRC_DIR="%cd%/_CPack_Packages/win64/ZIP/gdal-3.2.1-GNU-7.3/lib/site-packages/osgeo"
SET PYD_DST_DIR="%cd%\..\..\GDAL_win\lib\site-packages"
SET pyd_extension=pyd

if not exist %DST_DIR% (
	mkdir %DST_DIR%
)

xcopy %SRC_DIR% %DST_DIR% /s
for /r %GDAL_DLL_SRC_DIR% %%f in (*.%extension%) do @copy "%%f" %GDAL_DLL_DST_DIR%
for /r %DLL_SRC_DIR% %%f in (*.%extension%) do @copy "%%f" %DLL_DST_DIR%
for /r %PYD_SRC_DIR% %%f in (*.%pyd_extension%) do @copy "%%f" %PYD_DST_DIR%


if exist %GIF_DLL_FILE% (
    if not exist %REN_GIF_DLL_FILE% (
     ren %GIF_DLL_FILE% %REN_GIF_DLL_FILE%
    )
)