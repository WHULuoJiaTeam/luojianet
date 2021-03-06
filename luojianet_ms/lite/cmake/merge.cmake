function(merge_parser CL_SRC_DIR OUT_FILE_NAME)
    message(STATUS "Merge parser from ${CL_SRC_DIR} to ${OUT_FILE_NAME}")
    set(MAX_TIMESTAMP "000000000000.00")

    if(NOT EXISTS ${CL_SRC_DIR})
        return()
    endif()
    if(DEFINED MSLITE_COMPILE_TWICE AND NOT MSLITE_COMPILE_TWICE)
        return()
    endif()
    file(GLOB_RECURSE CL_LIST ${CL_SRC_DIR}/*.cc)
    list(SORT CL_LIST)
    set(out_file ${OUT_FILE_NAME})
    file(REMOVE ${out_file})
    file(WRITE ${out_file} "")
    foreach(file_path ${CL_LIST})
        file(TIMESTAMP ${file_path} CUR_TIMESTAMP "%Y%m%d%H%M.%S")
        string(COMPARE GREATER ${CUR_TIMESTAMP} ${MAX_TIMESTAMP} IS_GREATER)
        if(IS_GREATER)
            set(MAX_TIMESTAMP ${CUR_TIMESTAMP})
        endif()
        file(STRINGS ${file_path} READ_CC_CONTEXT NEWLINE_CONSUME NO_HEX_CONVERSION)
        file(APPEND ${out_file} ${READ_CC_CONTEXT})
    endforeach()
    execute_process(COMMAND touch -c -t ${MAX_TIMESTAMP} ${OUT_FILE_NAME})
endfunction()

function(merge_files SRC_DIR OUT_FILE_NAME EXCLUDE_FILES_PATTERN)
    message(STATUS "[merge_files] From ${SRC_DIR} to ${OUT_FILE_NAME}, exclude files: ${EXCLUDE_FILES_PATTERN}")
    set(MAX_TIMESTAMP "000000000000.00")

    file(GLOB_RECURSE SRC_LIST ${SRC_DIR}/*.cc)
    list(SORT SRC_LIST)

    file(REMOVE ${OUT_FILE_NAME})
    file(WRITE ${OUT_FILE_NAME} "")

    foreach(file_path ${SRC_LIST})
        if(EXISTS ${EXCLUDE_FILES_PATTERN})
            string(REGEX REPLACE ${EXCLUDE_FILES_PATTERN} "" file_path ${file_path})
        endif()
        if(EXISTS ${file_path})
            file(TIMESTAMP ${file_path} CUR_TIMESTAMP "%Y%m%d%H%M.%S")
            string(COMPARE GREATER ${CUR_TIMESTAMP} ${MAX_TIMESTAMP} IS_GREATER)
            if(IS_GREATER)
                set(MAX_TIMESTAMP ${CUR_TIMESTAMP})
            endif()
            file(STRINGS ${file_path} READ_CC_CONTEXT NEWLINE_CONSUME NO_HEX_CONVERSION)
            file(APPEND ${OUT_FILE_NAME} ${READ_CC_CONTEXT})
        else()
            message(STATUS "[merge_files] exclude file: ${file_path}${EXCLUDE_FILES_PATTERN}")
            continue()
        endif()
    endforeach()
    execute_process(COMMAND touch -c -t ${MAX_TIMESTAMP} ${OUT_FILE_NAME})
endfunction()
