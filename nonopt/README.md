# Accelerated Graph Processing
Bachelor thesis for Wouter de Bruijn (11239050) at the University of Amsterdam

## Instructions for building

This project requires the NVIDIA nvcc compiler, and either an offloading GCC/Clang C compiler or the NVIDIA PGI C compiler.
You can change your choice of compiler in the first few lines of the makefile.

## Usage

Enter "make" to build the entire project. The project requires an edgelist graph as input (with 1 being the first vertex).
Use "tester_pagerank_cuda" and "tester_pagerank_acc" to test the CUDA and OpenACC implementations on Pagerank. The BFS implementations
(tester_edge and tester_vertex) are not yet fully implemented.
