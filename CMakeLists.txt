cmake_minimum_required(VERSION 3.5.1)
project(tiger)

set(CMAKE_CXX_STANDARD 11)
set(tiger_link_libs "")
find_package(Protobuf REQUIRED)

if(PROTOBUF_FOUND)
    message(STATUS "protobuf library found")
    list(APPEND tiger_link_libs PUBLIC ${PROTOBUF_LIBRARIES})
else()
    message(FATAL_ERROR "protobuf library is needed but cant be found")
endif()

find_package(Boost REQUIRED filesystem)
list(APPEND tiger_link_libs PUBLIC ${Boost_LIBRARIES})

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

