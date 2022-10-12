cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(EEGraph LANGUAGES CXX CUDA)

# Variables and flags
# set(GPU_LIB "_cuda")
set(CPU_LIB "_cpp")
set(SSSP_PROJECT "sssp")
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "52;60;72")
endif()

# File definitions
file(GLOB sssp_main "${CMAKE_SOURCE_DIR}/src/sssp_main.cu")

file(GLOB cpp_source_files
  "${CMAKE_SOURCE_DIR}/src/argument_parsing.cpp"
  "${CMAKE_SOURCE_DIR}/src/graph.cpp"
  "${CMAKE_SOURCE_DIR}/src/tigr_utilities.cpp"
  "${CMAKE_SOURCE_DIR}/src/timer.cpp")

# Include directories
include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories(${GPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR/include})

# GPU Library
#add_library(${GPU_LIB} STATIC ${gpu_source_files})

# CPU Library
add_library(${CPU_LIB} STATIC ${cpp_source_files})

# Target compile feautes
#target_compile_features(${GPU_LIB} PUBLIC cxx_std_11)

# Executables
add_executable(${SSSP_PROJECT} ${sssp_main})

# Link libraries - SSSP
#target_link_libraries(${SSSP_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${SSSP_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${SSSP_PROJECT} PUBLIC -lnvidia-ml)

set_property(TARGET ${SSSP_PROJECT} PROPERTY CXX_STANDARD 11)