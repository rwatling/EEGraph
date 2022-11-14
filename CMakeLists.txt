cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(EEGraph LANGUAGES CXX CUDA)

##### Variables and flags
set(GPU_LIB "_cuda")
set(CPU_LIB "_cpp")
set(SSSP_PROJECT "sssp")
set(PR_PROJECT "pr")
set(SSWP_PROJECT "sswp")
set(BFS_PROJECT "bfs")
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "52;60;72")
endif()

##### FILES
file(GLOB sssp_main "${CMAKE_SOURCE_DIR}/src/sssp_main.cu")
file(GLOB pr_main "${CMAKE_SOURCE_DIR}/src/pr_main.cu")
file(GLOB sswp_main "${CMAKE_SOURCE_DIR}/src/sswp_main.cu")
file(GLOB bfs_main "${CMAKE_SOURCE_DIR}/src/bfs_main.cu")

file(GLOB cpp_source_files
  "${CMAKE_SOURCE_DIR}/src/argument_parsing.cpp"
  "${CMAKE_SOURCE_DIR}/src/graph.cpp"
  "${CMAKE_SOURCE_DIR}/src/utilities.cpp"
  "${CMAKE_SOURCE_DIR}/src/timer.cpp"
  "${CMAKE_SOURCE_DIR}/src/virtual_graph.cpp")

  file(GLOB gpu_source_files
    "${CMAKE_SOURCE_DIR}/src/sssp.cu"
    "${CMAKE_SOURCE_DIR}/src/pr.cu"
    "${CMAKE_SOURCE_DIR}/src/sswp.cu"
    "${CMAKE_SOURCE_DIR}/src/sssp_um.cu"
    "${CMAKE_SOURCE_DIR}/src/bfs.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu_utils.cu"
    "${CMAKE_SOURCE_DIR}/src/um_graph.cu"
    "${CMAKE_SOURCE_DIR}/src/um_virtual_graph.cu")

##### Include directories
include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${GPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

##### GPU Library
add_library(${GPU_LIB} STATIC ${gpu_source_files})

# Target compile feautes
target_compile_features(${GPU_LIB} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in ${GPU_LIB}
# to be built with -dc
set_target_properties(${GPU_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

##### CPU Library
add_library(${CPU_LIB} STATIC ${cpp_source_files})

##### Executables - SSSP
add_executable(${SSSP_PROJECT} ${sssp_main})

##### Link libraries - SSSP
target_link_libraries(${SSSP_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${SSSP_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${SSSP_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${SSSP_PROJECT} PROPERTY CXX_STANDARD 11)

##### Executables - PR
add_executable(${PR_PROJECT} ${pr_main})

##### Link libraries - PR
target_link_libraries(${PR_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${PR_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${PR_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${PR_PROJECT} PROPERTY CXX_STANDARD 11)

##### Executables - SSWP
add_executable(${SSWP_PROJECT} ${sswp_main})

##### Link libraries - SSWP
target_link_libraries(${SSWP_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${SSWP_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${SSWP_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${SSWP_PROJECT} PROPERTY CXX_STANDARD 11)

##### Executables - BFS
add_executable(${BFS_PROJECT} ${bfs_main})

##### Link libraries - BFS
target_link_libraries(${BFS_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${BFS_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${BFS_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${BFS_PROJECT} PROPERTY CXX_STANDARD 11)
