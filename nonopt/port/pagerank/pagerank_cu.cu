#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>

#include "pagerank_cu.h"
#include "../utils/pagerank_graph_utils.h"
#include "../highrestimer.h"

static void copy_graph_to_gpu(graph_cuda_t* graph) {
    for(uint32_t i = 0; i < graph->node_count; i++) {
        // Copy each node's in array
        checkCudaCall(cudaMalloc(&graph->nodes.host[i].in.device, graph->nodes.host[i].in_count * sizeof(uint32_t)));
        checkCudaCall(cudaMemcpyAsync(graph->nodes.host[i].in.device, graph->nodes.host[i].in.host,
                    graph->nodes.host[i].in_count * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // The outgoing edges are not useful in our case, so we don't copy them
    }

    checkCudaCall(cudaMalloc(&graph->nodes.device, graph->node_count * sizeof(node_cuda_t)));
    checkCudaCall(cudaMemcpyAsync(graph->nodes.device, graph->nodes.host,
                graph->node_count * sizeof(node_cuda_t), cudaMemcpyHostToDevice));
}

static void free_graph_from_gpu(graph_cuda_t* graph) {
    for(uint32_t i = 0; i < graph->node_count; i++) {
        checkCudaCall(cudaFree(graph->nodes.host[i].in.device));
    }

    checkCudaCall(cudaFree(graph->nodes.device));
}


__global__ void pagerank_do(graph_cuda_t* graph, pagerank_t* prev_rank, pagerank_t* next_rank) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < graph->node_count) {
        pagerank_t rank = 0;

        for(uint32_t j = 0; j < graph->nodes.device[i].in_count; j++) {
            uint32_t in_index = graph->nodes.device[i].in.device[j] - 1;
            rank += (prev_rank[in_index] / graph->nodes.device[in_index].out_count);
        }

        next_rank[i] = ((1.0f - PAGERANK_D) / graph->node_count) + (PAGERANK_D * rank);
    }
}

__global__ void pagerank_shift(pagerank_t* prev_rank, pagerank_t* next_rank, uint32_t maxcount) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < maxcount) {
        prev_rank[i] = next_rank[i];
    }
}

__device__ void pagerank_warp_max_reduce(volatile float* maxdata, int threadid) {
    // NOTE: The condition of the branches here should be evaluated at compile time
    // due to the macro CUDA_BLOCKSIZE. This should remove all the branches

    if(CUDA_BLOCKSIZE >= 64) maxdata[threadid] = fmaxf(maxdata[threadid], maxdata[threadid + 32]);
    if(CUDA_BLOCKSIZE >= 32) maxdata[threadid] = fmaxf(maxdata[threadid], maxdata[threadid + 16]);
    if(CUDA_BLOCKSIZE >= 16) maxdata[threadid] = fmaxf(maxdata[threadid], maxdata[threadid + 8]);
    if(CUDA_BLOCKSIZE >= 8) maxdata[threadid] = fmaxf(maxdata[threadid], maxdata[threadid + 4]);
    if(CUDA_BLOCKSIZE >= 4) maxdata[threadid] = fmaxf(maxdata[threadid], maxdata[threadid + 2]);
    if(CUDA_BLOCKSIZE >= 2) maxdata[threadid] = fmaxf(maxdata[threadid], maxdata[threadid + 1]);
}

/**
 * Highly optimised max-reduce function. Based on example taken from:
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
__global__ void pagerank_max_reduce(pagerank_t* prev_rank, pagerank_t* next_rank, float* block_max, uint32_t maxcount) {
    int i = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
    extern __shared__ float maxdata[];
    maxdata[threadIdx.x] = 0.0f;

    if(i < maxcount) {
        // Latency-hiding by using a single thread to already calculate a max on init
        float threshold_percentage1 = (fabsf(prev_rank[i] - next_rank[i]) / prev_rank[i]) * 100.0f;
        float threshold_percentage2 = (fabsf(prev_rank[i + blockDim.x] - next_rank[i + blockDim.x]) / prev_rank[i + blockDim.x]) * 100.0f;
        maxdata[threadIdx.x] = fmaxf(threshold_percentage1, threshold_percentage2);
        __syncthreads();

        // =====================vvv Unrolled for-loop vvv=======================
        // NOTE: The condition of the branches here should be evaluated at compile time
        // due to the macro CUDA_BLOCKSIZE. This should remove all the branches

        if(CUDA_BLOCKSIZE >= 512) {
            if(threadIdx.x < 256) {
                maxdata[threadIdx.x] = fmaxf(maxdata[threadIdx.x], maxdata[threadIdx.x + 256]);
            }
            __syncthreads();
        }

        if(CUDA_BLOCKSIZE >= 256) {
            if(threadIdx.x < 128) {
                maxdata[threadIdx.x] = fmaxf(maxdata[threadIdx.x], maxdata[threadIdx.x + 128]);
            }
            __syncthreads();
        }

        if(CUDA_BLOCKSIZE >= 128) {
            if(threadIdx.x < 64) {
                maxdata[threadIdx.x] = fmaxf(maxdata[threadIdx.x], maxdata[threadIdx.x + 64]);
            }
            __syncthreads();
        }

        if(threadIdx.x < 32) {
            pagerank_warp_max_reduce(maxdata, threadIdx.x);
        }

        // =====================^^^ Unrolled for-loop ^^^=======================

        // Get all the results from all the blocks
        if(threadIdx.x == 0) {
            block_max[blockIdx.x] = maxdata[0];
        }

        __syncthreads();
    }
}

pagerank_t* do_pagerank_cu(graph_cuda_t* graph) {
    // As we unrolled for-loops in the GPU reduce function, we put a hard
    // limit on the possible block sizes at 512. This should not be an issue,
    // as CUDA block sizes should be a maximum of 512 anyway
    assert(CUDA_BLOCKSIZE <= 512);

    fputs("Starting host -> device memory transfers\n", stderr);

    highrestimer_t start_time = get_highrestime();

    unsigned int block_count = (graph->node_count / CUDA_BLOCKSIZE) + 1;

    pagerank_t* ranks = init_pagerank(graph->node_count);
    pagerank_t* ranks_device = NULL;
    pagerank_t* ranks_device_next = NULL;

    // Fills in the device pointers in the graph
    copy_graph_to_gpu(graph);
    graph_cuda_t* graph_device = NULL;

    float* max_threshold = (float*) malloc(block_count / 2 * sizeof(float));

    for(size_t i = 0; i < block_count / 2; i++) {
        max_threshold[i] = 0.0f;
    }

    float* max_threshold_device = NULL;

    // Allocate and init GPU memory
    checkCudaCall(cudaMalloc(&ranks_device, graph->node_count * sizeof(pagerank_t)));
    checkCudaCall(cudaMalloc(&ranks_device_next, graph->node_count * sizeof(pagerank_t)));

    checkCudaCall(cudaMalloc(&graph_device, sizeof(graph_cuda_t)));

    checkCudaCall(cudaMalloc(&max_threshold_device, block_count / 2 * sizeof(float)));

    checkCudaCall(cudaMemcpyAsync(ranks_device, ranks, graph->node_count * sizeof(pagerank_t), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpyAsync(max_threshold_device, max_threshold, block_count / 2 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpyAsync(graph_device, graph, sizeof(graph_cuda_t), cudaMemcpyHostToDevice));

    checkCudaCall(cudaDeviceSynchronize());

    fputs("Memory transfer done, starting computation\n", stderr);
    highrestimer_t start_time_no_mem = get_highrestime();
    float prev_threshold = 0.0f;

    // To keep track of any perpetually stalling graphs
    unsigned int num_same = 0;
    do {
        pagerank_do<<<block_count, CUDA_BLOCKSIZE>>>
                   (graph_device, ranks_device, ranks_device_next);

        pagerank_max_reduce<<<block_count / 2, CUDA_BLOCKSIZE, CUDA_BLOCKSIZE * sizeof(float)>>>
                   (ranks_device, ranks_device_next, max_threshold_device, graph->node_count);

        // Copy max threshold to host
        checkCudaCall(cudaMemcpy(max_threshold, max_threshold_device, block_count / 2 * sizeof(float), cudaMemcpyDeviceToHost));

        pagerank_shift<<<block_count, CUDA_BLOCKSIZE>>>
                   (ranks_device, ranks_device_next, graph->node_count);

        // Find the maximum of all the GPU blocks
        for(size_t i = 1; i < block_count / 2; i++) {
            max_threshold[0] = MAX(max_threshold[0], max_threshold[i]);
        }

        // Pagerank sometimes has some weird precision issues which stops the program from
        // terminating. We solve this by stopping pagerank when the thresholds dont change
        if((fabsf(max_threshold[0] - prev_threshold) < 0.0001f)) {
            if(++num_same > 10) {
                break;
            }
        } else {
            num_same = 0;
            prev_threshold = max_threshold[0];
        }
    } while(max_threshold[0] > PAGERANK_THRESHOLD);
    checkCudaCall(cudaDeviceSynchronize());

    fputs("Pagerank converged, freeing GPU memory\n", stderr);

    // Get the rank data from the GPU
    checkCudaCall(cudaMemcpy(ranks, ranks_device, graph->node_count * sizeof(pagerank_t), cudaMemcpyDeviceToHost));

    highrestimer_t end_time_no_mem = get_highrestime();

    // Free GPU memory
    checkCudaCall(cudaFree(ranks_device));
    checkCudaCall(cudaFree(ranks_device_next));
    checkCudaCall(cudaFree(max_threshold_device));
    checkCudaCall(cudaFree(graph_device));
    free_graph_from_gpu(graph);

    checkCudaCall(cudaDeviceSynchronize());

    highrestimer_t end_time = get_highrestime();

    double runtime = highrestime_diff(start_time, end_time);
    double runtime_no_mem = highrestime_diff(start_time_no_mem, end_time_no_mem);

    printf("%f %f %f\n", runtime, runtime_no_mem, runtime - runtime_no_mem);

    return ranks;
}
