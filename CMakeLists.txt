cmake_policy(VERSION 3.20)
cmake_minimum_required(VERSION 3.0)
project(EE-Graph LANGUAGES CXX CUDA VERSION 1.0)
############## DEPENDENCIES ##############
find_package(CUDA 9 REQUIRED)

message("-- CUDA Version: ${CUDA_VERSION_STRING}")

# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

# CUB
include_directories(${PROJECT_SOURCE_DIR}/deps/cub)

# JSON
include_directories(${PROJECT_SOURCE_DIR}/deps/json)
############## BUILD ##############

set(EXTRA_LIBS gflags pthread ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -std=c++14)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")

message("-- Host Compiler: ${CUDA_HOST_COMPILER}")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

# hybrid apps
#cuda_add_executable(hybrid_pr src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp #src/groute/graphs/csr_graph.cpp
#        samples/hybrid_pr/hybrid_pr.cu
#        samples/hybrid_pr/hybrid_pr_host.cpp
#        samples/hybrid_pr/main.cpp)
#target_link_libraries(hybrid_pr ${EXTRA_LIBS})

add_executable(${PROJECT_NAME} src/main.cpp)

add_executable(gtools src/utils/parser.cpp src/utils/utils.cpp
        src/tools/gtools.cpp)
target_link_libraries(gtools gflags)
