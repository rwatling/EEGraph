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

int bfs_async_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int bfs_sync_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int cc_async_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int cc_sync_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdge> graph);

int pr_async_driver(SubwayArgumentParser arguments, GraphPR<OutEdge> graph);

int pr_sync_driver(SubwayArgumentParser arguments, GraphPR<OutEdge> graph);

int sswp_async_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);

int sswp_sync_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);

int sswp_async_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);

int sswp_sync_driver(SubwayArgumentParser arguments, SubwayGraph<OutEdgeWeighted> graph);