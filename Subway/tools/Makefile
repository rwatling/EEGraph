
CC=g++
NC=nvcc
CFLAGS=-std=c++14 -O3
NFLAGS=-arch=sm_60 -lnvidia-ml

SHARED=../shared

all: converter

converter: converter.cpp
	$(CC) converter.cpp -o converter $(CFLAGS)
			
clean:
	rm -f converter
