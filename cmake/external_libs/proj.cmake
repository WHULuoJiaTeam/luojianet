if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors_OSGeo/PROJ/repository/archive/7.1.1.zip")
    set(MD5 "b04cfa8997d623b27675f2f5965ad5d0")
else()
    set(REQ_URL "https://github.com/OSGeo/PROJ/releases/download/6.3.1/proj-7.1.1.tar.gz")
    set(MD5 "c44c694cf569a74880e5fbac566d54d6")
endif()

set(proj_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
# set(proj_CXXFLAGS)
# if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	# set(proj_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 \
	  # -O2")
# else()
	# set(proj_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC \
	  # -D_FORTIFY_SOURCE=2 -O2 -dl")
	# set(sqlite_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack, -dl")
# endif()
# MESSAGE(FATAL_ERROR "sqlite_INC: ${sqlite_INC}")
# set(proj_USE_STATIC_LIBS ON)
set(CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DPROJ_TESTS=OFF -DENABLE_CURL=ON
    # -DCMAKE_PREFIX_PATH=${sqlite_INC}/..
    -DSQLITE3_INCLUDE_DIR=${sqlite_INC} -DSQLITE3_LIBRARY=${sqlite3_LIB}
	-DHAVE_DLFCN_H=1
	)
luojianet_ms_add_pkg(proj
        VER 7.1.0
        LIBS proj
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION ${CMAKE_OPTION}
        )

include_directories(${proj_INC})
add_library(luojianet_ms::proj ALIAS proj::proj)



