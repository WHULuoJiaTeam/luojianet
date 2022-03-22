# "stub"  usage:

## Description

- File libge_compiler.so ,libgraph.so are used in IR build application interface.

# Attention

- Don't link other library except libge_compiler.so ,libgraph.so, as they may be changed in the future.

# Usage

## Compile:   compile  the application invoking the IR build API.

Makefile:

'''

ATC_INCLUDE_DIR := $(ASCEND_PATH)/atc/include
OPP_INCLUDE_DIR := $(ASCEND_PATH)/opp/op_proto/built-in/inc
LOCAL_MODULE_NAME := ir_build
CC := g++
CFLAGS := -std=c++11 -g -Wall
SRCS := $(wildcard $(LOCAL_DIR)/main.cpp)
INCLUDES := -I $(ASCEND_OPP_PATH)/op_proto/built-in/inc \
            -I $(ATC_INCLUDE_DIR)/graph \
            -I $(ATC_INCLUDE_DIR)/ge \

LIBS := -L ${ASCEND_PATH}/atc/lib64/stub \
    -lgraph \
    -lge_compiler
ir_build:
    mkdir -p out
    $(CC) $(SRCS) $(INCLUDES) $(LIBS) $(CFLAGS) -o ./out/$(LOCAL_MODULE_NAME)
clean:
    rm -rf out

'''
make

## Run the application after set the LD_LIBRARY_PATH to include the real path of the library which locates in the directory of atc/lib64

export LD_LIBRARY_PATH= $(ASCEND_PATH)/atc/lib64
 -  ./ ir_build
