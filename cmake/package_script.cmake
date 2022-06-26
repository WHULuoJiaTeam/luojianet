# find exec
find_package(Git)
if(NOT GIT_FOUND)
    message("No git found.")
    return()
endif()
set(GIT ${GIT_EXECUTABLE})

# set path
set(MS_ROOT_DIR ${CPACK_CMAKE_SOURCE_DIR})
set(MS_PACK_ROOT_DIR ${CPACK_PACK_ROOT_DIR})

set(PYTHON ${CPACK_PYTHON_EXE})
set(PYTHON_VERSION ${CPACK_PYTHON_VERSION})

if(NOT (PYTHON_VERSION MATCHES "3.9" OR PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7"))
    message(FATAL_ERROR "FIND PYTHON VERSION ${PYTHON_VERSION} BUT CAN NOT MATCH PYTHON VERSION 3.9 OR 3.8 OR 3.7")
endif()

# set package file name
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(PYTHON_VERSION MATCHES "3.9")
        set(PY_TAGS "cp39-cp39")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(PY_TAGS "cp38-cp38")
    elseif(PYTHON_VERSION MATCHES "3.7")
        set(PY_TAGS "cp37-cp37m")
    else()
        message("Could not find 'Python 3.9' OR 'Python 3.8' or 'Python 3.7'")
        return()
    endif()
    string(TOLOWER linux_${CMAKE_HOST_SYSTEM_PROCESSOR} PLATFORM_TAG)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    if(PYTHON_VERSION MATCHES "3.9")
        set(PY_TAGS "cp39-cp39")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(PY_TAGS "cp38-cp38")
    elseif(PYTHON_VERSION MATCHES "3.7")
        set(PY_TAGS "cp37-cp37m")
    else()
        message("Could not find 'Python 3.9' OR 'Python 3.8' or 'Python 3.7'")
        return()
    endif()
    string(REGEX MATCH "[0-9]+.[0-9]+" MACOSX_SDK_VERSION "${CPACK_OSX_DEPLOYMENT_TARGET}")
    string(REPLACE "." "_" MACOSX_PACKAGE_VERSION "${MACOSX_SDK_VERSION}")
    string(TOLOWER macosx_${MACOSX_PACKAGE_VERSION}_${CMAKE_HOST_SYSTEM_PROCESSOR} PLATFORM_TAG)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(PYTHON_VERSION MATCHES "3.9")
        set(PY_TAGS "cp39-cp39")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(PY_TAGS "cp38-cp38")
    elseif(PYTHON_VERSION MATCHES "3.7")
        set(PY_TAGS "cp37-cp37m")
    else()
        message("Could not find 'Python 3.9' OR 'Python 3.8' or 'Python 3.7'")
        return()
    endif()
    set(PLATFORM_TAG "win_amd64")
else()
    message(FATAL_ERROR "other platform: ${CMAKE_SYSTEM_NAME}")
endif()

# get git commit id
set(GIT_COMMIT_ID "")
execute_process(
    COMMAND ${GIT} log --format='[sha1]:%h,[branch]:%d' --abbrev=8 -1
    OUTPUT_VARIABLE GIT_COMMIT_ID
    WORKING_DIRECTORY ${MS_ROOT_DIR}
    ERROR_QUIET)
string(REPLACE " " "" GIT_COMMIT_ID ${GIT_COMMIT_ID})

set(ENV{BACKEND_POLICY} ${CPACK_MS_BACKEND})
set(ENV{BACKEND_TARGET} ${CPACK_MS_TARGET})
set(ENV{MS_PACKAGE_NAME} ${CPACK_MS_PACKAGE_NAME})
set(ENV{COMMIT_ID} ${GIT_COMMIT_ID})

file(GLOB DEBUG_SYM
    ${MS_PACK_ROOT_DIR}/mindspore/*.so
    ${MS_PACK_ROOT_DIR}/mindspore/lib/*.so
)

file(GLOB DEBUG_STRIP_SYM
    ${MS_PACK_ROOT_DIR}/mindspore/*.so
    ${MS_PACK_ROOT_DIR}/mindspore/lib/*.so*
)
list(REMOVE_ITEM DEBUG_STRIP_SYM ${MS_PACK_ROOT_DIR}/mindspore/lib/libmindspore_aicpu_kernels.so)

set(CMAKE_OBJCOPY $ENV{CROSS_COMPILE}objcopy)
set(CMAKE_STRIP $ENV{CROSS_COMPILE}strip)

if(CPACK_ENABLE_SYM_FILE)
    foreach(schema ${DEBUG_SYM})
        execute_process(
            COMMAND ${CMAKE_OBJCOPY} "--only-keep-debug" ${schema} ${schema}.sym
            WORKING_DIRECTORY ${MS_PACK_ROOT_DIR}
    )
    endforeach()
endif()

if("${CPACK_CMAKE_BUILD_TYPE}" STREQUAL "Release")
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(CMAKE_STRIP_ARGS "-x")
    else()
        set(CMAKE_STRIP_ARGS "")
    endif()
    foreach(schema ${DEBUG_STRIP_SYM})
    execute_process(
        COMMAND ${CMAKE_STRIP} ${CMAKE_STRIP_ARGS} ${schema}
        WORKING_DIRECTORY ${MS_PACK_ROOT_DIR}
    )
    endforeach()
endif()

file(GLOB DEBUG_SYM_FILE
    ${MS_PACK_ROOT_DIR}/mindspore/*.sym
    ${MS_PACK_ROOT_DIR}/mindspore/lib/*.sym
)

if(CPACK_ENABLE_SYM_FILE)
    file(MAKE_DIRECTORY ${MS_ROOT_DIR}/debug_info)
    file(COPY ${DEBUG_SYM_FILE} DESTINATION ${MS_ROOT_DIR}/debug_info/)
    file(REMOVE_RECURSE ${DEBUG_SYM_FILE})
endif()

execute_process(
    COMMAND ${PYTHON} ${MS_ROOT_DIR}/setup.py "bdist_wheel"
    WORKING_DIRECTORY ${MS_PACK_ROOT_DIR}
)

# finally
set(PACKAGE_NAME ${CPACK_MS_PACKAGE_NAME})
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    string(REPLACE "-" "_" PACKAGE_NAME ${PACKAGE_NAME})
    execute_process(
        COMMAND chmod -R 700 ${MS_PACK_ROOT_DIR}/mindspore/
        COMMAND chmod -R 700 ${MS_PACK_ROOT_DIR}/${PACKAGE_NAME}.egg-info/
    )
endif()

file(GLOB WHL_FILE ${MS_PACK_ROOT_DIR}/dist/*.whl)
get_filename_component(ORIGIN_FILE_NAME ${WHL_FILE} NAME)
string(REPLACE "-" ";" ORIGIN_FILE_NAME ${ORIGIN_FILE_NAME})
list(GET ORIGIN_FILE_NAME 1 VERSION)
set(NEW_FILE_NAME ${PACKAGE_NAME}-${VERSION}-${PY_TAGS}-${PLATFORM_TAG}.whl)
file(RENAME ${WHL_FILE} ${MS_PACK_ROOT_DIR}/${NEW_FILE_NAME})
file(REMOVE_RECURSE ${MS_ROOT_DIR}/output)
file(MAKE_DIRECTORY ${MS_ROOT_DIR}/output)
file(COPY ${MS_PACK_ROOT_DIR}/${NEW_FILE_NAME} DESTINATION ${MS_ROOT_DIR}/output/)

file(SHA256 ${MS_ROOT_DIR}/output/${NEW_FILE_NAME} SHA256_VAR)
file(WRITE ${MS_ROOT_DIR}/output/${NEW_FILE_NAME}.sha256 ${SHA256_VAR} " " ${NEW_FILE_NAME})
set(CMAKE_TAR $ENV{CROSS_COMPILE}tar)
if(CPACK_ENABLE_SYM_FILE)
    file(MAKE_DIRECTORY ${MS_ROOT_DIR}/output/${PACKAGE_NAME}-${VERSION}-${PY_TAGS}-${PLATFORM_TAG})
    file(COPY ${MS_ROOT_DIR}/debug_info/ DESTINATION
        ${MS_ROOT_DIR}/output/${PACKAGE_NAME}-${VERSION}-${PY_TAGS}-${PLATFORM_TAG}/)
    execute_process(COMMAND
        ${CMAKE_COMMAND} -E ${CMAKE_TAR} cfv
        ${MS_ROOT_DIR}/output/${PACKAGE_NAME}-${VERSION}-${PY_TAGS}-${PLATFORM_TAG}.zip
        ${MS_ROOT_DIR}/output/${PACKAGE_NAME}-${VERSION}-${PY_TAGS}-${PLATFORM_TAG}/ --format=zip
        WORKING_DIRECTORY ${MS_ROOT_DIR})
    file(REMOVE_RECURSE ${MS_ROOT_DIR}/debug_info)
    file(REMOVE_RECURSE ${MS_ROOT_DIR}/output/${PACKAGE_NAME}-${VERSION}-${PY_TAGS}-${PLATFORM_TAG})
endif()
