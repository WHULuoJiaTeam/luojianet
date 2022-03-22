if (HAVE_GTEST)
    return()
endif()

include(ExternalProject)

if ((${CMAKE_INSTALL_PREFIX} STREQUAL /usr/local) OR
    (${CMAKE_INSTALL_PREFIX} STREQUAL "C:/Program Files (x86)/ascend"))
    set(CMAKE_INSTALL_PREFIX ${METADEF_DIR}/output CACHE STRING "path for install()" FORCE)
    message(STATUS "No install prefix selected, default to ${CMAKE_INSTALL_PREFIX}.")
endif()

if (METADEF_PB_PKG)
    set(REQ_URL "${METADEF_PB_PKG}/libs/ge_gtest/release-1.8.1.tar.gz")
    set(MD5 "")
elseif (ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/googletest/repository/archive/release-1.8.1.tar.gz")
    set(MD5 "")
else()
    set(REQ_URL "https://github.com/google/googletest/archive/release-1.8.1.tar.gz")
    set(MD5 "")
endif ()

set (gtest_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set (gtest_CFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
ExternalProject_Add(gtest_build
                    URL ${REQ_URL}
                    TLS_VERIFY OFF
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_FLAGS=${gtest_CXXFLAGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/gtest <SOURCE_DIR>
                -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_MACOSX_RPATH=TRUE -Dgtest_disable_pthreads=ON
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE 
)

set(GTEST_PKG_DIR ${CMAKE_INSTALL_PREFIX}/gtest)

file(MAKE_DIRECTORY ${GTEST_PKG_DIR}/include)

add_library(gtest SHARED IMPORTED)

set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${GTEST_PKG_DIR}/lib/libgtest.so
)

add_library(gtest_main SHARED IMPORTED)

set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION ${GTEST_PKG_DIR}/lib/libgtest_main.so
)

target_include_directories(gtest INTERFACE ${GTEST_PKG_DIR}/include)
target_include_directories(gtest_main INTERFACE ${GTEST_PKG_DIR}/include)

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(FILES ${GTEST_PKG_DIR}/lib/libgtest.so ${GTEST_PKG_DIR}/lib/libgtest_main.so OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR})

add_dependencies(gtest gtest_build)

#set(HAVE_GFLAGS TRUE CACHE BOOL "gflags build add")
set(HAVE_GTEST TRUE)
