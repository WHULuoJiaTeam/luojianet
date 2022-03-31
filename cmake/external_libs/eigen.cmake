set(Eigen3_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Eigen3_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")


set(REQ_URL "https://gitee.com/mizhangwhuer/eigen/attach_files/1014163/download/eigen-3.3.9.tar.gz")
set(MD5 "d41d8cd98f00b204e9800998ecf8427e")


luojianet_ms_add_pkg(Eigen3
        VER 3.3.9
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DBUILD_TESTING=OFF)
find_package(Eigen3 3.3.9 REQUIRED ${MS_FIND_NO_DEFAULT_PATH})
include_directories(${Eigen3_INC})
include_directories(${EIGEN3_INCLUDE_DIR})
set_property(TARGET Eigen3::Eigen PROPERTY IMPORTED_GLOBAL TRUE)
add_library(luojianet_ms::eigen ALIAS Eigen3::Eigen)
