cmake_minimum_required(VERSION 3.5.1)
project(tiger)

set(CMAKE_CXX_STANDARD 11)

find_package(Protobuf REQUIRED)
if(PROTOBUF_FOUND)
    message(STATUS "protobuf library found")
else()
    message(FATAL_ERROR "protobuf library is needed but cant be found")
endif()


PROTOBUF_GENERATE_CPP( PROTO_SRCS PROTO_HDRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/tiger.proto)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/*.cc
    )

link_libraries(glog pthread protobuf)
add_library(tiger ${cpp_srcs} ${PROTO_SRCS} ${PROTO_HDRS})

add_subdirectory(test)

