#ifndef __PAGERANK_CUDA_UTILS_C__
#define __PAGERANK_CUDA_UTILS_C__

#include "pagerank_graph_utils.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <limits.h>
#include <inttypes.h>

#include "../main.h"

#define CUDA_BLOCKSIZE (128)
#define checkKernelError()
#define checkCudaCall(function) (function)

typedef struct uint32_devpointer {
    uint32_t* host;
    uint32_t* device;
} uint32_dp_t;

typedef struct node_cuda {
    uint32_t out_count;
    uint32_t __out_alloced;
    uint32_dp_t out;

    uint32_t in_count;
    uint32_t __in_alloced;
    uint32_dp_t in;
} node_cuda_t;

typedef struct node_devpointer {
    node_cuda_t* host;
    node_cuda_t* device;
} node_dp_t;

typedef struct graph_cuda {
    uint32_t node_count;
    uint32_t __alloced_space;
    node_dp_t nodes;
} graph_cuda_t;

static int realloc_graph_node_array_cuda(graph_cuda_t* graph, unsigned int new_size) {
    uint32_t starting_node_count = graph->node_count;

    if(new_size > graph->__alloced_space) {
        uint32_t new_alloced_space = MAX((1.3333 * graph->__alloced_space), graph->__alloced_space + 10);
        new_alloced_space = MAX(new_alloced_space, new_size + 10);

        graph->nodes.host = (node_cuda_t*) realloc(graph->nodes.host, new_alloced_space * sizeof(node_cuda_t));

        if(graph->nodes.host == NULL) {
            fputs("Memory allocation failure\n", stderr);
            return 0;
        }

        graph->__alloced_space = new_alloced_space;
    }

    // Modify the node_count after the realloc error check because
    // realloc only changes the nodes array size when there is no error
    graph->node_count = new_size;

    for(uint32_t i = starting_node_count; i < new_size; i++) {
        graph->nodes.host[i].in_count = 0;
        graph->nodes.host[i].__in_alloced = 1;
        graph->nodes.host[i].in.host = (uint32_t*) malloc(sizeof(uint32_t));

        graph->nodes.host[i].out.host = (uint32_t*) malloc(sizeof(uint32_t));
        graph->nodes.host[i].__out_alloced = 1;
        graph->nodes.host[i].out_count = 0;
    }

    return 1;
}

static int node_add_connection_cuda(graph_cuda_t* graph, unsigned int from, unsigned int to) {
    void* realloc_result;

    graph->nodes.host[from - 1].out_count++;
    graph->nodes.host[to - 1].in_count++;

    // Check if there is enough space for the new outgoing connection
    if(graph->nodes.host[from - 1].__out_alloced < graph->nodes.host[from - 1].out_count) {
        uint32_t new_alloc_size = MAX((1.3333 * graph->nodes.host[from - 1].__out_alloced), graph->nodes.host[from - 1].__out_alloced + 10);
        graph->nodes.host[from - 1].__out_alloced = new_alloc_size;

        realloc_result = realloc(graph->nodes.host[from - 1].out.host, graph->nodes.host[from - 1].__out_alloced * sizeof(uint32_t));

        if(realloc_result == NULL) {
            fputs("Memory allocation failure\n", stderr);
            return 0;
        }

        graph->nodes.host[from - 1].out.host = (uint32_t*) realloc_result;
    }

    // Check if there is enough space for the new incoming connection
    if(graph->nodes.host[to - 1].__in_alloced < graph->nodes.host[to - 1].in_count) {
        uint32_t new_alloc_size = MAX((1.3333 * graph->nodes.host[to - 1].__in_alloced), graph->nodes.host[to - 1].__in_alloced + 10);
        graph->nodes.host[to - 1].__in_alloced = new_alloc_size;

        realloc_result = realloc(graph->nodes.host[to - 1].in.host, graph->nodes.host[to - 1].__in_alloced * sizeof(uint32_t));

        if(realloc_result == NULL) {
            fputs("Memory allocation failure\n", stderr);
            return 0;
        }
        graph->nodes.host[to - 1].in.host = (uint32_t*) realloc_result;
    }

    // Add the connections
    graph->nodes.host[from - 1].out.host[graph->nodes.host[from - 1].out_count - 1] = to;
    graph->nodes.host[to - 1].in.host[graph->nodes.host[to - 1].in_count - 1] = from;

    return 1;
}

static int graph_handle_sinks_cuda(graph_cuda_t* graph) {
    fputs("Finding and handling sinks\n", stderr);

    for(uint32_t i = 0; i < graph->node_count; i++) {
        if(graph->nodes.host[i].out_count == 0) {
            for(uint32_t j = 0; j < graph->nodes.host[i].in_count; j++) {
                if(!node_add_connection_cuda(graph, i + 1, graph->nodes.host[i].in.host[j])) {
                    return EXIT_FAILURE;
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

static void shrink_graph_cuda_to_size(graph_cuda_t* graph) {
    fputs("Shrinking graph to proper memory size\n", stderr);

    graph->nodes.host = (node_cuda_t*) realloc(graph->nodes.host, graph->node_count * sizeof(node_cuda_t));
    graph->__alloced_space = graph->node_count;

    for(uint32_t i = 0; i < graph->node_count; i++) {
        node_cuda_t* node = &graph->nodes.host[i];
        node->in.host = (uint32_t*) realloc(node->in.host, node->in_count * sizeof(uint32_t));
        node->__in_alloced = node->in_count;
        node->out.host = (uint32_t*) realloc(node->out.host, node->out_count * sizeof(uint32_t));
        node->__out_alloced = node->out_count;
    }
}

static void free_graph_pagerank_cuda(graph_cuda_t* tofree) {
    for(uint32_t i = 0; i < tofree->node_count; i++) {
        free(tofree->nodes.host[i].in.host);
        free(tofree->nodes.host[i].out.host);
    }

    free(tofree->nodes.host);
    free(tofree);
}

static graph_cuda_t* parse_input_file_pagerank_cuda(char* dataset_path) {
    // Initialise the graph structure
    graph_cuda_t* graph = (graph_cuda_t*) malloc(sizeof(graph_cuda_t));

    graph->node_count = 0;
    graph->__alloced_space = 1;
    graph->nodes.host = (node_cuda_t*) malloc(sizeof(node_cuda_t));

    FILE* dataset = fopen(dataset_path, "r");

    if(dataset == NULL) {
        // Couldn't open dataset
        fputs("Couldn't open dataset file!\n", stderr);
        free(graph->nodes.host);
        free(graph);
        return NULL;
    }

    char line[BUF_SIZE];
    // Read dataset line by line and parse the result
    fputs("Reading input graph...\n", stderr);
    unsigned long long linenr = 0;
    unsigned long long edgecount = 0;
    int is_zero_indexed = 0;

    do {
        linenr++;

        char* result = fgets(line, BUF_SIZE, dataset);

        if(result == NULL) {
            break;
        }

        unsigned int from = UINT_MAX;
        unsigned int to = UINT_MAX;
        unsigned int weight = 0;
        unsigned long timestamp = 0;

        sscanf(line, "%u %u %u %lu\n", &from, &to, &weight, &timestamp);

        if(from != UINT_MAX && to != UINT_MAX) {
            if((from == 0 || to == 0) && is_zero_indexed == 0) {
                is_zero_indexed = 1;
                fprintf(stderr, "Zero-indexed graph (line %llu)\n", linenr);
            }

            if(is_zero_indexed) {
                from++;
                to++;
            }

            // Check if we have enough space for nodes
            if((MAX(from, to)) > graph->node_count) {
                if(!realloc_graph_node_array_cuda(graph, MAX(from, to))) {
                    fputs("Couldn't acquire memory to expand graph node array\n", stderr);
                    free_graph_pagerank_cuda(graph);
                    fclose(dataset);
                    return NULL;
                }
            }

            if(!node_add_connection_cuda(graph, from, to)) {
                fputs("Couldn't add node connection\n", stderr);
                free_graph_pagerank_cuda(graph);
                fclose(dataset);
                return NULL;
            }

            edgecount++;
        } else {
            fprintf(stderr, "Discarding invalid line at %llu\n", linenr);
        }
    } while(line != NULL);

    if(graph_handle_sinks_cuda(graph) != EXIT_SUCCESS) {
        fputs("Couldn't add node connection\n", stderr);
        free_graph_pagerank_cuda(graph);
        fclose(dataset);
        return NULL;
    }

    shrink_graph_cuda_to_size(graph);

    fprintf(stderr, "Graph contains %" PRIu32 " nodes and %llu edges\n", graph->node_count, edgecount);

    fclose(dataset);
    return graph;
}

#ifdef __CUDACC__
/* Utility function, use to do error checking.

   Use this function like this:
   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:
   checkCudaCall(cudaGetLastError());
*/
static void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) {
        fputs("CUDA error: ", stderr);
        fprintf(stderr, "%s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

#endif

#endif
