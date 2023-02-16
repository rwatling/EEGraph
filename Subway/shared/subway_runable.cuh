#include "subway_globals.hpp"
#include "../../include/timer.hpp"
#include "../../include/utilities.hpp"
#include "subway_argument_parsing.cuh"
#include "subway_graph.cuh"
#include "subgraph.cuh"
#include "partitioner.cuh"
#include "subgraph_generator.cuh"
#include "gpu_error_check.cuh"
#include "gpu_kernels.cuh"
#include "nvmlClass.cuh"

int subway_bfs_async(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int subway_bfs_sync(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int subway_cc_async(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int subway_cc_sync(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int subway_pr_async(SubwayArgumentParser arguments, GraphPR<OutEdge> graph);

int subway_pr_sync(SubwayArgumentParser arguments, GraphPR<OutEdge> graph);

int subway_sswp_async(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);

int subway_sswp_sync(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);

int subway_sswp_async(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);

int subway_sswp_sync(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);