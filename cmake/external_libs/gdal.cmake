if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/GDAL/repository/archive/v3.0.4.zip")
    set(MD5 "abc86c3a8e0a1b74f427e05406087f82")
    set(SUB_DIR gdal)
else()
    set(REQ_URL "https://github.com/OSGeo/gdal/releases/download/v3.0.4/gdal-3.0.4.tar.gz")
    set(MD5 "c6bbb5caca06e96bd97a32918e0aa9aa")
endif()


get_filename_component(PROJ_DIR ${proj_INC} DIRECTORY)
get_filename_component(PROJ_LIB_PATH ${proj_LIB} DIRECTORY)

set(gdal_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized \
    -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2 -I${proj_INC}")
set(gdal_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -I${proj_INC}")
set(gdal_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack -L${PROJ_LIB_PATH}")

luojianet_ms_add_pkg(gdal
        VER 3.0.4
        LIBS gdal
        URL ${REQ_URL}
        MD5 ${MD5}
        SUB_DIR ${SUB_DIR}
        CONFIGURE_COMMAND ./configure
            --with-libtiff=internal --with-rename-internal-libtiff-symbols
            --with-geotiff=internal --with-rename-internal-libgeotiff-symbols
            --without-libtool
            --disable-all-optional-drivers
            --without-netcdf
            --without-hdf4
            --without-hdf5
            --without-pcraster
            --without-pcidsk
            --without-lerc
            --without-gnm
            --without-gif
            --without-jpeg
            --without-png
            --without-crypto
            --without-cryptopp
            --without-curl
            --without-zstd
            --without-expat
            --without-geos
        )
include_directories(${gdal_INC})
add_library(luojianet_ms::gdal ALIAS gdal::gdal)
