if (HAVE_PROTOC)
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

set(protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2")
set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
ExternalProject_Add(protoc_build
                    URL ${REQ_URL}
                    TLS_VERIFY OFF
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} -Dprotobuf_WITH_ZLIB=OFF -Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=${protobuf_CXXFLAGS} -DCMAKE_CXX_LDFLAGS=${protobuf_LDFLAGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/protoc <SOURCE_DIR>/cmake
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE
)

set(PROTOC_PKG_DIR ${CMAKE_INSTALL_PREFIX}/protoc)

set(protoc_EXECUTABLE ${PROTOC_PKG_DIR}/${CMAKE_INSTALL_BINDIR}/protoc)

function(protobuf_generate comp c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
        return()
    endif()
    set(${c_var})
    set(${h_var})
    set(_add_target FALSE)

    foreach(file ${ARGN})
        if("${file}" STREQUAL "TARGET")
            set(_add_target TRUE)
            continue()
        endif()

        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        get_filename_component(parent_subdir ${file_dir} NAME)

        if("${parent_subdir}" STREQUAL "proto")
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
        else()
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
        endif()
        list(APPEND ${c_var} "${proto_output_path}/${file_name}.pb.cc")
        list(APPEND ${h_var} "${proto_output_path}/${file_name}.pb.h")

        add_custom_command(
                OUTPUT "${proto_output_path}/${file_name}.pb.cc" "${proto_output_path}/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
                COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
                COMMAND ${protoc_EXECUTABLE} -I${file_dir} --cpp_out=${proto_output_path} ${abs_file}
                DEPENDS protoc_build ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    if(_add_target)
        add_custom_target(
            ${comp} DEPENDS ${${c_var}} ${${h_var}}
        )
    endif()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)

endfunction()

function(protobuf_generate_py comp py_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: protobuf_generate_py() called without any proto files")
        return()
    endif()
    set(${py_var})
    set(_add_target FALSE)

    foreach(file ${ARGN})
        if("${file}" STREQUAL "TARGET")
            set(_add_target TRUE)
            continue()
        endif()

        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        get_filename_component(parent_subdir ${file_dir} NAME)

        if("${parent_subdir}" STREQUAL "proto")
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
        else()
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
        endif()
        list(APPEND ${py_var} "${proto_output_path}/${file_name}_pb2.py")

        add_custom_command(
                OUTPUT "${proto_output_path}/${file_name}_pb2.py"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
                COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
                COMMAND ${protoc_EXECUTABLE} -I${file_dir} --python_out=${proto_output_path} ${abs_file}
                DEPENDS protoc_build ${abs_file}
                COMMENT "Running PYTHON protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    if(_add_target)
        add_custom_target(
            ${comp} DEPENDS ${${py_var}}
        )
    endif()

    set_source_files_properties(${${py_var}} PROPERTIES GENERATED TRUE)
    set(${py_var} ${${py_var}} PARENT_SCOPE)

endfunction()

#set(HAVE_PROTOC TRUE CACHE BOOL "protoc build add")
set(HAVE_PROTOC TRUE)
