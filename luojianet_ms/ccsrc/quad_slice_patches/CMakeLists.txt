PROJECT(geobject)
CMAKE_MINIMUM_REQUIRED(VERSION 3.18.0)

if(MINGW OR UNIX)
	set(CMAKE_CXX_FLAGS
		  "${CMAKE_CXX_FLAGS} -Wno-unused-result")
endif()


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)


#gdal
if(ENABLE_RS)
    remove_definitions(-DHAVE_STDLIB_H)
    if(WIN32)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/include)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/include/gdal)
        # message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/third_party/GDAL/include/gdal")
        LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/lib)
    elseif(UNIX)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/include)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/include/gdal)
        # message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/third_party/GDAL/include/gdal")
        LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/lib)
    endif()
endif()

#pybind11
include_directories(${pybind11_INC})
# find_package(pybind11 REQUIRED)
# set_property(TARGET pybind11::module PROPERTY IMPORTED_GLOBAL TRUE)
# add_library(luojianet_ms::pybind11_module ALIAS pybind11::module)

# FIND_PACKAGE(OpenCV REQUIRED)

# Find PythonLibs
# FIND_PACKAGE(PythonLibs REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

FILE(GLOB HED_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*)
FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)


pybind11_add_module(${PROJECT_NAME} ${HED_FILES} ${SRC_FILES})

if(UNIX)
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE luojianet_ms::pybind11_module libgdal.so luojianet_ms::opencv_core
											  luojianet_ms::opencv_imgcodecs luojianet_ms::opencv_imgproc ${PYTHON_LIBRARIES})
elseif(MINGW)
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE luojianet_ms::pybind11_module libgdal.dll.a luojianet_ms::opencv_core
											  luojianet_ms::opencv_imgcodecs luojianet_ms::opencv_imgproc ${PYTHON_LIBRARIES})
else()
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE luojianet_ms::pybind11_module libgdal luojianet_ms::opencv_core
											  luojianet_ms::opencv_imgcodecs luojianet_ms::opencv_imgproc ${PYTHON_LIBRARIES})
endif()