PROTOBUF_GENERATE_CPP( PROTO_SRCS PROTO_HDRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/tiger.proto)

add_library(prototiger ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(prototiger ${PROTOBUF_LIBRARIES})
list(APPEND tiger_link_libs PUBLIC prototiger)

if(HAVE_CUDA)
    file(GLOB_RECURSE cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    cuda_include_directories(${CMAKE_SOURCE_DIR}/include)
    cuda_add_library(gpu ${cuda_srcs})
    CUDA_ADD_CUBLAS_TO_TARGET(gpu)
    list(APPEND tiger_link_libs PUBLIC gpu)
endif()


file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cpp
    )
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})


add_library(tiger ${cpp_srcs})
target_link_libraries(tiger ${tiger_link_libs})
