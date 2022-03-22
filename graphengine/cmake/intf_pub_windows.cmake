
add_library(intf_pub INTERFACE)

target_compile_options(intf_pub INTERFACE
	-Wall
    -fPIC
    $<IF:$<STREQUAL:${OS_TYPE},centos>,-fstack-protector-all,-fstack-protector-strong>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
)
target_compile_definitions(intf_pub INTERFACE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    OS_TYPE=WIN64
    WIN64=1
    LINUX=0
	$<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
	$<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
)
target_link_options(intf_pub INTERFACE
    $<$<CONFIG:Release>:-Wl,--build-id=none>
)
target_link_directories(intf_pub INTERFACE
)
target_link_libraries(intf_pub INTERFACE
)
