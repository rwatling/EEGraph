cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(EEGraph LANGUAGES CXX CUDA)

##### Variables and flags
set(GPU_LIB "_cuda")
set(CPU_LIB "_cpp")
set(SSSP_PROJECT "sssp-main")
set(PR_PROJECT "pr-main")
set(SSWP_PROJECT "sswp-main")
set(BFS_PROJECT "bfs-main")
set(CC_PROJECT "cc-main")
set(MAIN_PROJECT "main")
set(MAIN_LARGE_PROJECT "main-large")
set(GRAPH_INFO_PROJECT "graph-info")
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "52;60;72")
endif()

##### FILES
file(GLOB sssp_main "${CMAKE_SOURCE_DIR}/src/sssp_main.cu")
file(GLOB pr_main "${CMAKE_SOURCE_DIR}/src/pr_main.cu")
file(GLOB sswp_main "${CMAKE_SOURCE_DIR}/src/sswp_main.cu")
file(GLOB bfs_main "${CMAKE_SOURCE_DIR}/src/bfs_main.cu")
file(GLOB cc_main "${CMAKE_SOURCE_DIR}/src/cc_main.cu")
file(GLOB main "${CMAKE_SOURCE_DIR}/src/main.cu")
file(GLOB main_large "${CMAKE_SOURCE_DIR}/src/main_large.cu")
file(GLOB graph_info "${CMAKE_SOURCE_DIR}/src/graph_info.cpp")

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
    "${CMAKE_SOURCE_DIR}/src/bfs.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu_utils.cu"
    "${CMAKE_SOURCE_DIR}/src/um_graph.cu"
    "${CMAKE_SOURCE_DIR}/src/um_virtual_graph.cu"
    "${CMAKE_SOURCE_DIR}/src/cc.cu"
    "${CMAKE_SOURCE_DIR}/src/eegraph.cu")

##### Include directories
include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${GPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

##### GPU Library
add_library(${GPU_LIB} STATIC ${gpu_source_files})

# Target compile feautes
target_compile_features(${GPU_LIB} PUBLIC cxx_std_14)

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
set_property(TARGET ${SSSP_PROJECT} PROPERTY CXX_STANDARD 14)

##### Executables - PR
add_executable(${PR_PROJECT} ${pr_main})

##### Link libraries - PR
target_link_libraries(${PR_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${PR_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${PR_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${PR_PROJECT} PROPERTY CXX_STANDARD 14)

##### Executables - SSWP
add_executable(${SSWP_PROJECT} ${sswp_main})

##### Link libraries - SSWP
target_link_libraries(${SSWP_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${SSWP_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${SSWP_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${SSWP_PROJECT} PROPERTY CXX_STANDARD 14)

##### Executables - BFS
add_executable(${BFS_PROJECT} ${bfs_main})

##### Link libraries - BFS
target_link_libraries(${BFS_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${BFS_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${BFS_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${BFS_PROJECT} PROPERTY CXX_STANDARD 14)

##### Executables - CC
add_executable(${CC_PROJECT} ${cc_main})

##### Link libraries - CC
target_link_libraries(${CC_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${CC_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${CC_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${CC_PROJECT} PROPERTY CXX_STANDARD 14)

##### Executables - MAIN
add_executable(${MAIN_PROJECT} ${main})

##### Link libraries - MAIN PROJECT
target_link_libraries(${MAIN_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${MAIN_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${MAIN_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${MAIN_PROJECT} PROPERTY CXX_STANDARD 14)

##### Executables - MAIN LARGE
add_executable(${MAIN_LARGE_PROJECT} ${main_large})

##### Link libraries - MAIN LARGE PROJECT
target_link_libraries(${MAIN_LARGE_PROJECT} PRIVATE ${GPU_LIB})
target_link_libraries(${MAIN_LARGE_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${MAIN_LARGE_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${MAIN_LARGE_PROJECT} PROPERTY CXX_STANDARD 14)

##### Executables - Graph Info
add_executable(${GRAPH_INFO_PROJECT} ${graph_info})

##### Link libraries - Graph Info PROJECT
target_link_libraries(${GRAPH_INFO_PROJECT} PRIVATE ${CPU_LIB})
set_property(TARGET ${GRAPH_INFO_PROJECT} PROPERTY CXX_STANDARD 14)
