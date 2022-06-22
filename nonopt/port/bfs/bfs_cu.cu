#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <time.h>

#include "bfs_cu.h"
#include "../utils/bfs_graph_utils.h"
#include "../highrestimer.h"
double total = 0;
double count = 0;

__global__ void bfs_search(edge_t* edges, result_t* results, edge_count_t edge_count) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < edge_count) {
        uint32_t origin_index = edges[i].from - 1;
        uint32_t destination_index = edges[i].to - 1;

        if(results[origin_index].state == node_toprocess) {
            if(results[destination_index].state == node_unvisited) {
                results[destination_index].state = node_reachable;
                results[destination_index].depth = results[origin_index].depth + 1;
            }
        }
    }
}

__global__ void bfs_update_state(result_t* results, uint32_t node_count, int* was_updated) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < node_count) {
        switch(results[i].state) {
            case node_unvisited:
            case node_visited:
                break;
            case node_reachable:
                results[i].state = node_toprocess;
                *(was_updated) = 1;
                break;
            case node_toprocess:
                results[i].state = node_visited;
                *(was_updated) = 1;
                break;
        }
    }
}


result_t* do_bfs_cuda(graph_edge_t* graph) {
    fputs("Starting CUDA BFS\n", stderr);

    highrestimer_t start_time, end_time, start_time_no_mem, end_time_no_mem;

    int* was_updated = (int *) malloc(sizeof(int));

    result_t* results = (result_t *) malloc(graph->node_count * sizeof(result_t));

    for(uint32_t i = 0; i < graph->node_count; i++) {
        results[i].state = node_unvisited;
        results[i].depth = 0;
    }

    results[0].state = node_toprocess;

    start_time = get_highrestime();

    // Copy data to GPU
    fputs("Copying data to GPU\n", stderr);
    edge_t* edges_device = NULL;
    result_t* results_device = NULL;
    int* was_updated_device = NULL;

    checkCudaCall(cudaMalloc(&edges_device, graph->edge_count * sizeof(edge_t)));
    checkCudaCall(cudaMalloc(&results_device, graph->node_count * sizeof(result_t)));
    checkCudaCall(cudaMalloc(&was_updated_device, sizeof(int)));

    checkCudaCall(cudaMemcpy(edges_device, graph->edges, graph->edge_count * sizeof(edge_t), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(results_device, results, graph->node_count * sizeof(result_t), cudaMemcpyHostToDevice));

    // Do BFS
    checkCudaCall(cudaDeviceSynchronize());
    fputs("Done copying, starting BFS\n", stderr);
    start_time_no_mem = get_highrestime();
    do {
        checkCudaCall(cudaMemset(was_updated_device, 0, sizeof(int)));

        bfs_search<<<(graph->edge_count / CUDA_BLOCKSIZE) + 1, CUDA_BLOCKSIZE>>>
                    (edges_device, results_device, graph->edge_count);


        bfs_update_state<<<(graph->node_count / CUDA_BLOCKSIZE) + 1, CUDA_BLOCKSIZE>>>
                    (results_device, graph->node_count, was_updated_device);


        checkCudaCall(cudaMemcpy(was_updated, was_updated_device, sizeof(int), cudaMemcpyDeviceToHost));

    } while(*(was_updated) == 1);

    checkCudaCall(cudaDeviceSynchronize());
    end_time_no_mem = get_highrestime();
    fputs("BFS done, copying results back\n", stderr);

    // Copy results back
    checkCudaCall(cudaMemcpy(results, results_device, graph->node_count * sizeof(result_t), cudaMemcpyDeviceToHost));

    // Delete data from GPU
    checkCudaCall(cudaFree(edges_device));
    checkCudaCall(cudaFree(results_device));
    checkCudaCall(cudaFree(was_updated_device));
    checkCudaCall(cudaDeviceSynchronize());

    end_time = get_highrestime();
    free(was_updated);

    fputs("Done copying results\n", stderr);

    double runtime = highrestime_diff(start_time, end_time);
    double runtime_no_mem = highrestime_diff(start_time_no_mem, end_time_no_mem);

    printf("%f %f %f\n", runtime, runtime_no_mem, runtime - runtime_no_mem);

    return results;
}
