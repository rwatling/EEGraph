#ifndef SUBGRAPH_GENERATOR_HPP
#define SUBGRAPH_GENERATOR_HPP


#include "subway_globals.hpp"
#include "subway_graph.cuh"
#include "subgraph.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>

template <class E>
class SubgraphGenerator
{
private:

public:
	unsigned int *activeNodesLabeling;
	unsigned int *activeNodesDegree;
	unsigned int *prefixLabeling;
	unsigned int *prefixSumDegrees;
	unsigned int *d_activeNodesLabeling;
	unsigned int *d_activeNodesDegree;
	unsigned int *d_prefixLabeling;
	unsigned int *d_prefixSumDegrees;
	SubgraphGenerator(SubwayGraph<E> &graph);
	SubgraphGenerator(GraphPR<E> &graph);
	void generate(SubwayGraph<E> &graph, Subgraph<E> &subgraph);
	void generate(GraphPR<E> &graph, Subgraph<E> &subgraph, float acc);
};

#endif	//	SUBGRAPH_GENERATOR_HPP



