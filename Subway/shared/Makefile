CC=g++
NC=nvcc
CFLAGS=-std=c++14 -O3
NFLAGS=-arch=sm_60 -lnvidia-ml


all: timer.o subway_argument_parsing.o subway_graph.o subgraph.o partitioner.o subgraph_generator.o gpu_kernels.o subway_utilities.o


timer.o: timer.cpp
	$(CC) -c timer.cpp -o timer.o $(CFLAGS)
	
subway_argument_parsing.o: subway_argument_parsing.cu
	$(NC) -c subway_argument_parsing.cu -o subway_argument_parsing.o $(CFLAGS) $(NFLAGS)

subway_graph.o: subway_graph.cu
	$(NC) -c subway_graph.cu -o subway_graph.o $(CFLAGS) $(NFLAGS)

subgraph.o: subgraph.cu
	$(NC) -c subgraph.cu -o subgraph.o $(CFLAGS) $(NFLAGS)

partitioner.o: partitioner.cu
	$(NC) -c partitioner.cu -o partitioner.o $(CFLAGS) $(NFLAGS)

subgraph_generator.o: subgraph_generator.cu
	$(NC) -c subgraph_generator.cu -o subgraph_generator.o $(CFLAGS) $(NFLAGS)
	
gpu_kernels.o: gpu_kernels.cu
	$(NC) -c gpu_kernels.cu -o gpu_kernels.o $(CFLAGS) $(NFLAGS)

subway_utilities.o: subway_utilities.cpp
	$(CC) -c subway_utilities.cpp -o subway_utilities.o $(CFLAGS)


clean:
	rm *.o
