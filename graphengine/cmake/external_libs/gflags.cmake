if (HAVE_GFLAGS)
    return()
endif()

include(ExternalProject)

if ((${CMAKE_INSTALL_PREFIX} STREQUAL /usr/local) OR
    (${CMAKE_INSTALL_PREFIX} STREQUAL "C:/Program Files (x86)/ascend"))
    set(CMAKE_INSTALL_PREFIX ${GE_CODE_DIR}/output CACHE STRING "path for install()" FORCE)
    message(STATUS "No install prefix selected, default to ${CMAKE_INSTALL_PREFIX}.")
endif()

if (GE_PB_PKG)
    set(REQ_URL "${GE_PB_PKG}/libs/gflags/v2.2.2.tar.gz")
    set(MD5 "1a865b93bacfa963201af3f75b7bd64c")
else()
    if (ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/gflags/repository/archive/v2.2.2.tar.gz")
        set(MD5 "")
    else()
        set(REQ_URL "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz")
        set(MD5 "1a865b93bacfa963201af3f75b7bd64c")
    endif ()
endif ()

set (gflags_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=ascend_private")
ExternalProject_Add(gflags_build
                    URL ${REQ_URL}
                    #URL /home/txd/workspace/linux_cmake/pkg/protobuf-3.8.0.tar.gz
                    #SOURCE_DIR ${GE_CODE_DIR}/../../third_party/gflags/src/gflags-2.2.2 
                    TLS_VERIFY OFF
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_FLAGS=${gflags_CXXFLAGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/gflags <SOURCE_DIR>
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE 
)

set(GFLAGS_PKG_DIR ${CMAKE_INSTALL_PREFIX}/gflags)

add_library(gflags_static STATIC IMPORTED)

set_target_properties(gflags_static PROPERTIES
                      IMPORTED_LOCATION ${GFLAGS_PKG_DIR}/lib/libgflags.a
)

add_library(gflags INTERFACE)
target_include_directories(gflags INTERFACE ${GFLAGS_PKG_DIR}/include)
target_link_libraries(gflags INTERFACE gflags_static)

add_dependencies(gflags gflags_build)

#set(HAVE_GFLAGS TRUE CACHE BOOL "gflags build add")
set(HAVE_GFLAGS TRUE)
