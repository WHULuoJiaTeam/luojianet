if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/c-ares/repository/archive/cares-1_15_0.tar.gz")
    set(MD5 "459876e53724a2c87ed2876312b96aed")
else()
    set(REQ_URL "https://github.com/c-ares/c-ares/releases/download/cares-1_15_0/c-ares-1.15.0.tar.gz")
    set(MD5 "d2391da274653f7643270623e822dff7")
endif()

mindspore_add_pkg(c-ares
        VER 1.15.0
        LIBS cares
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DCARES_SHARED:BOOL=OFF
        -DCARES_STATIC:BOOL=ON
        -DCARES_STATIC_PIC:BOOL=ON
        -DHAVE_LIBNSL:BOOL=OFF
        PATCHES ${TOP_DIR}/third_party/patch/c-ares/CVE-2021-3672.patch)

include_directories(${c-ares_INC})
add_library(mindspore::cares ALIAS c-ares::cares)
