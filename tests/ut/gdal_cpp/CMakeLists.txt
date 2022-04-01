PROJECT(GDAL_Testing)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wno-deprecated -Wno-unknown-pragmas -Wno-reorder -Wno-unused-function -Wno-char-subscripts -Wno-switch -Wno-unused-but-set-parameter -Wno-unused-value -Wno-unused-local-typedefs -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-ignored-qualifiers -Wno-cast-qual -Wno-parentheses -Wno-return-type -fPIC")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

if(ENABLE_RS)
	if(WIN32)
		INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/include)
		INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/include/gdal)
		# message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/third_party/GDAL/include/gdal")
		LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/lib)
	elseif(UNIX)
		INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/include)
		INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/include/gdal)
		message("Link dir: ${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/lib")
		LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/lib)
	endif()
endif()

FILE(GLOB HED_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)

ADD_EXECUTABLE(${PROJECT_NAME} ${HED_FILES} ${SRC_FILES})

if(UNIX)
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} libgdal)
else()
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} libgdal.dll.a)
endif()