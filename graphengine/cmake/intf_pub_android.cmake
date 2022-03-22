
add_library(intf_pub INTERFACE)

target_compile_options(intf_pub INTERFACE 	
	-Wall 
    -fPIC 
    -fstack-protector-strong
)
target_compile_definitions(intf_pub INTERFACE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0> 
	$<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
    $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
    WIN64=1
    LINUX=0
)
target_link_options(intf_pub INTERFACE
	-Wl,-z,relro
	-Wl,-z,now
	-Wl,-z,noexecstack
    $<$<CONFIG:Release>:-Wl,--build-id=none>    
)
target_link_directories(intf_pub INTERFACE
)

add_library(intf_ccec INTERFACE)
target_compile_options(intf_ccec INTERFACE 	
    -mcpu=cortex-a73
    --target=aarch64-linux-android29
    --sysroot=${HCC_PATH}/../sysroot
    -L${HCC_PATH}/../lib/gcc/aarch64-linux-android/4.9.x
	-Wall 
    -fPIC 
    -fstack-protector-strong
)
target_compile_definitions(intf_ccec INTERFACE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0> 
	$<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
	$<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
)

target_link_options(intf_ccec INTERFACE
    -mcpu=cortex-a73
    --target=aarch64-linux-android29
    --sysroot=${HCC_PATH}/../sysroot
    -L${HCC_PATH}/../lib/gcc/aarch64-linux-android/4.9.x
    -Wl,-cce-host-android
	-Wl,-z,relro
	-Wl,-z,now
	-Wl,-z,noexecstack
    $<$<CONFIG:Release>:-Wl,--build-id=none> 
)

