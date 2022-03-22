if (HAVE_PROTOBUF_STATIC)
    return()
endif()

include(ExternalProject)
include(GNUInstallDirs)
#set(CMAKE_INSTALL_PREFIX ${GE_CODE_DIR}/output)

if ((${CMAKE_INSTALL_PREFIX} STREQUAL /usr/local) OR
    (${CMAKE_INSTALL_PREFIX} STREQUAL "C:/Program Files (x86)/ascend"))
    set(CMAKE_INSTALL_PREFIX ${GE_CODE_DIR}/output CACHE STRING "path for install()" FORCE)
    message(STATUS "No install prefix selected, default to ${CMAKE_INSTALL_PREFIX}.")
endif()

if(GE_PB_PKG)
    set(REQ_URL "${GE_PB_PKG}/libs/protobuf/v3.13.0.tar.gz")
else()
    if (ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.13.0.tar.gz")
        set(MD5 "f4489cb88922ad9c58cbe3308d59cee5")
    else()
        set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz")
        set(MD5 "1a6274bc4a65b55a6fa70e264d796490")
    endif ()
endif()

set(protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -Dgoogle=ascend_private")
set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
set(PROTOBUF_STATIC_PKG_DIR ${CMAKE_INSTALL_PREFIX}/protobuf_static)
ExternalProject_Add(protobuf_static_build
                    URL ${REQ_URL}
                    TLS_VERIFY OFF
                    CONFIGURE_COMMAND ${CMAKE_COMMAND}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                    -DCMAKE_LINKER=${CMAKE_LINKER}
                    -DCMAKE_AR=${CMAKE_AR}
                    -DCMAKE_RANLIB=${CMAKE_RANLIB}
                    -Dprotobuf_WITH_ZLIB=OFF
                    -DLIB_PREFIX=ascend_
                    -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_CXX_FLAGS=${protobuf_CXXFLAGS} -DCMAKE_CXX_LDFLAGS=${protobuf_LDFLAGS} -DCMAKE_INSTALL_PREFIX=${PROTOBUF_STATIC_PKG_DIR} <SOURCE_DIR>/cmake
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE
)
include(GNUInstallDirs)

add_library(ascend_protobuf_static_lib STATIC IMPORTED)

set_target_properties(ascend_protobuf_static_lib PROPERTIES
                      IMPORTED_LOCATION ${PROTOBUF_STATIC_PKG_DIR}/${CMAKE_INSTALL_LIBDIR}/libascend_protobuf.a
)

add_library(ascend_protobuf_static INTERFACE)
target_include_directories(ascend_protobuf_static INTERFACE ${PROTOBUF_STATIC_PKG_DIR}/include)
target_link_libraries(ascend_protobuf_static INTERFACE ascend_protobuf_static_lib)
if (ENABLE_D OR ENABLE_ACL OR ENABLE_MS_TESTCASES)
include_directories(${PROTOBUF_STATIC_PKG_DIR}/include)
endif ()

add_dependencies(ascend_protobuf_static protobuf_static_build)

set(HAVE_PROTOBUF_STATIC TRUE)
