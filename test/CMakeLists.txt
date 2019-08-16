cmake_minimum_required(VERSION 3.5.1)
file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} ${tiger_test_libs})
endforeach(source)

message(STATUS "libs is ${tiger_test_libs}")

file(GLOB cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
foreach(source ${cuda_srcs})
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    get_filename_component(name ${source} NAME_WE)
    cuda_add_executable(${name} ${source})
    target_link_libraries(${name} gpu)
endforeach(source)

