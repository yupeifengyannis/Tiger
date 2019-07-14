cmake_minimum_required(VERSION 3.5.1)
project(tiger)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 11)
set(tiger_link_libs "")

option(CPU_ONLY "only use cpu" OFF)
option(HAVE_CUDA "have_cuda" ON)
if(CPU_ONLY)
    add_definitions(-DCPU_ONLY)
    option(HAVE_CUDA "have cuda" OFF)
endif()

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

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "opencv found")
    list(APPEND tiger_link_libs PUBLIC ${OpenCV_LIBRARIES})
else()
    message(FATAL_ERROR "opencv is needed but can't be found")
endif()


find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "cuda found")
    include_directories(CUDA_INCLUDE_DIRS)
else()
    message(FATAL_ERROR "cuda is needed but can't be found")
endif()

PROTOBUF_GENERATE_CPP( PROTO_SRCS PROTO_HDRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/tiger.proto)

add_library(prototiger ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(prototiger ${PROTOBUF_LIBRARIES})
list(APPEND tiger_link_libs PUBLIC prototiger)

file(GLOB_RECURSE cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
# 需要制定cuda工程的头文件目录，不然编译老是找不到头文件
# nvcc编译的时候需要用到c++11的特性，所以这里添加相应的flag
if(HAVE_CUDA)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    cuda_add_library(gpu ${cuda_srcs})
    list(APPEND tiger_link_libs PUBLIC gpu)
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_BINARY_DIR})
file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layers/*.cpp
    )

link_libraries(glog gflags leveldb)
add_library(tiger ${cpp_srcs})
target_link_libraries(tiger ${tiger_link_libs})

add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(sample/mnist)
add_subdirectory(sample/bdd100k)
