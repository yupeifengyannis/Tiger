cmake_minimum_required(VERSION 3.5.1)
project(tiger)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.cc
    )


message(${cpp_srcs})

link_libraries(glog)
add_library(tiger ${cpp_srcs})

add_subdirectory(test)






