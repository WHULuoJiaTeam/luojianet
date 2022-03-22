if (HAVE_JSON)
    return()
endif()

include(ExternalProject)

set(JSON_SRC_DIR ${CMAKE_BINARY_DIR}/opensrc/json/include)
if (GE_PB_PKG)
    set(REQ_URL "${GE_PB_PKG}/libs/ge_nlohmann_json/include.zip")
    set(MD5 "0dc903888211db3a0f170304cd9f3a89")
    set(JSON_INCLUDE_DIR ${JSON_SRC_DIR})
else()
    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip")
    set(MD5 "0dc903888211db3a0f170304cd9f3a89")
    set(JSON_INCLUDE_DIR ${JSON_SRC_DIR})
endif ()
ExternalProject_Add(json_build
                    URL ${REQ_URL}
                    #URL /home/txd/workspace/cloud_code/pkg/include.zip
                    SOURCE_DIR  ${JSON_SRC_DIR}
                    TLS_VERIFY OFF
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE 
)


add_library(json INTERFACE)
target_include_directories(json INTERFACE ${JSON_INCLUDE_DIR})
add_dependencies(json json_build)

#set(HAVE_JSON TRUE CACHE BOOL "json build add")
set(HAVE_JSON TRUE)
