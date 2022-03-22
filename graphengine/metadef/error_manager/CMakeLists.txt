set(SRC_LIST
    "error_manager.cc"
)

############ liberror_manager.so ############
add_library(error_manager SHARED ${SRC_LIST})

target_compile_options(error_manager PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-Werror -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_include_directories(error_manager PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../inc>
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc>
)

target_link_libraries(error_manager PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    static_mmpa
    -Wl,--no-as-needed
    c_sec
    slog
    -Wl,--as-needed
    json
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

############ liberror_manager.a ############
add_library(error_manager_static STATIC ${SRC_LIST})

target_include_directories(error_manager_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../inc>
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc>
)

target_compile_definitions(error_manager_static PRIVATE
    LOG_CPP
)

target_compile_options(error_manager_static PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-Werror -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_link_libraries(error_manager_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    json
    c_sec
)

set_target_properties(error_manager_static PROPERTIES
    OUTPUT_NAME error_manager
)

set_target_properties(error_manager PROPERTIES
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,liberror_manager,error_manager>
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

if(TARGET_SYSTEM_NAME STREQUAL "Windows")
install(TARGETS error_manager error_manager_static OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
else()
install(TARGETS error_manager error_manager_static OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
)
endif()
