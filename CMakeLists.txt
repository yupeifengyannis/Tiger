cmake_minimum_required(VERSION 3.5.1)
project(tiger)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp)
file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.cc)


link_libraries(glog)
add_library(tiger ${cpp_srcs})







