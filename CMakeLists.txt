cmake_minimum_required(VERSION 3.5.1)
project(tiger)

set(CMAKE_CXX_STANDARD 11)
set(tiger_link_libs "")

option(CPU_ONLY "只用cpu" ON)


find_package(Protobuf REQUIRED)
if(PROTOBUF_FOUND)
    message(STATUS "protobuf library found")
    list(APPEND tiger_link_libs PUBLIC ${PROTOBUF_LIBRARIES})
    include_directories(PRTOBUF_INCLUDE_DIRS)
else()
    message(FATAL_ERROR "protobuf library is needed but can't be found")
endif()

find_package(Boost REQUIRED filesystem)
if(Boost_FOUND)
    message(STATUS "boost filesystem found")
    list(APPEND tiger_link_libs PUBLIC ${Boost_LIBRARIES})
    include_directories(Boost_INCLUDE_DIRS)
else()
    message(FATAL_ERROR "boost filesystem is needed but can't be found")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "cuda found")
    include_directories(CUDA_INCLUDE_DIRS)
else()
    message(FATAL_ERROR "cuda is needed but can't be found")
endif()

file(GLOB_RECURSE cuda_headers *.h *.hpp *.cuh)
file(GLOB_RECURSE cuda_srcs *.cu)
cuda_add_library(gpu ${cuda_srcs} ${cuda_headers})

list(APPEND tiger_link_libs PUBLIC gpu)


PROTOBUF_GENERATE_CPP( PROTO_SRCS PROTO_HDRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/tiger.proto)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR})
file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.cc
    )

link_libraries(glog)
add_library(tiger ${cpp_srcs} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(tiger ${tiger_link_libs})

add_subdirectory(test)

