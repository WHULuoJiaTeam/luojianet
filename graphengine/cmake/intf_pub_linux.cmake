if (HAVE_PUB)
    return()
endif()

add_library(intf_pub INTERFACE)

target_compile_options(intf_pub INTERFACE 	
    -Wall 
    -fPIC 
    $<IF:$<STREQUAL:${CMAKE_SYSTEM_NAME},centos>,-fstack-protector-all,-fstack-protector-strong>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>    
)
target_compile_definitions(intf_pub INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
    $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>   
    WIN64=1
    LINUX=0
    LOG_CPP
)
target_link_options(intf_pub INTERFACE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-Wl,--build-id=none>    
)
target_link_directories(intf_pub INTERFACE
)
target_link_libraries(intf_pub INTERFACE 
    -lpthread
)

#set(HAVE_PUB TRUE CACHE BOOL "pub add")
set(HAVE_PUB TRUE)
