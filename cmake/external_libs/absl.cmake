if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/abseil-cpp.git")
    set(GIT_TAG "openEuler-22.03-LTS")
    set(MD5 "06c9554d9f9776bb1a4b848ca206c0af")
    set(ABSL_SRC "${TOP_DIR}/build/mindspore/_deps/absl-src")
    __download_pkg_with_git(absl ${GIT_REPOSITORY} ${GIT_TAG} ${MD5})
    execute_process(COMMAND tar -xf ${ABSL_SRC}/abseil-cpp-20210324.2.tar.gz --strip-components 1 -C ${ABSL_SRC})
else()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/abseil-cpp/repository/archive/20210324.2.tar.gz")
    set(MD5 "e0b585398b89ee92a10d8c68ef7eed48")
else()
    set(REQ_URL "https://github.com/abseil/abseil-cpp/archive/20210324.2.tar.gz")
    set(MD5 "e0b585398b89ee92a10d8c68ef7eed48")
endif()
endif()

if(NOT ENABLE_GLIBCXX)
    set(absl_CXXFLAGS "${absl_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mindspore_add_pkg(absl
        VER 20210324.2
        LIBS absl_strings absl_throw_delegate absl_raw_logging_internal absl_int128 absl_bad_optional_access
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
        -DCMAKE_CXX_STANDARD=11
        )

include_directories(${absl_INC})

add_library(mindspore::absl_strings ALIAS absl::absl_strings)
add_library(mindspore::absl_throw_delegate ALIAS absl::absl_throw_delegate)
add_library(mindspore::absl_raw_logging_internal ALIAS absl::absl_raw_logging_internal)
add_library(mindspore::absl_int128 ALIAS absl::absl_int128)
add_library(mindspore::absl_bad_optional_access ALIAS absl::absl_bad_optional_access)
