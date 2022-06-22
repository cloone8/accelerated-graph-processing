#ifndef __BFS_GRAPH_UTILS_H__
#define __BFS_GRAPH_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <limits.h>

#include "../main.h"

#define CUDA_BLOCKSIZE (128)
#define checkKernelError() (cuda_check(cudaGetLastError()))
#define checkCudaCall(function) (cuda_check(function))

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef UINT64_MAX
    typedef uint64_t edge_count_t;
    #define PRIedgecount PRIu64
#else
    typedef uint32_t edge_count_t;
    #define PRIedgecount PRIu32
#endif

typedef struct edge {
    uint32_t from;
    uint32_t to;
} edge_t;

typedef struct graph_edge {
    uint32_t node_count;
    edge_count_t edge_count;
    edge_count_t __alloced_space;
    edge_t* edges;
} graph_edge_t;

typedef enum {
    node_unvisited = 0, // Node has not yet been visited and is not reachable
    node_reachable = 1, // Node is reachable, but has not been visited
    node_visited = 2, // Node has been visited
    node_toprocess = 3 // Special state for when a node has been visited. Added to avoid race conditions.
                        // This state will be converted to node_visited in a following loop
} result_state_t;

// General result struct
typedef struct result {
    result_state_t state;
    uint32_t depth;
} result_t;

static void shrink_graph_edge_to_size(graph_edge_t* graph) {
    fputs("Shrinking graph to proper memory size\n", stderr);

    graph->edges = (edge_t *) realloc(graph->edges, graph->edge_count * sizeof(edge_t));
    graph->__alloced_space = graph->edge_count;
}

static int graph_edge_add_connection(graph_edge_t* graph, unsigned int from, unsigned int to) {
    // Check if enough space has been allocated
    if(graph->edge_count + 1 > graph->__alloced_space) {
        edge_count_t new_space = MAX((1.5 * graph->__alloced_space), (graph->__alloced_space + 200));

        void* alloc_result = realloc(graph->edges, new_space * sizeof(edge_t));

        if(alloc_result == NULL) {
            return EXIT_FAILURE;
        }

        graph->edges = (edge_t *) alloc_result;
        graph->__alloced_space = new_space;
    }

    graph->edge_count++;
    if(MAX(from, to) > graph->node_count) {
        graph->node_count = MAX(from, to);
    }

    graph->edges[graph->edge_count - 1].from = from;
    graph->edges[graph->edge_count - 1].to = to;

    return EXIT_SUCCESS;
}

static void free_graph_edge(graph_edge_t* tofree) {
    free(tofree->edges);
    free(tofree);
}

static graph_edge_t* parse_input_file_edge(char* dataset_path) {
    // Initialise the graph structure
    graph_edge_t* graph = (graph_edge_t *) malloc(sizeof(graph_edge_t));

    graph->node_count = 0;
    graph->edge_count = 0;
    graph->__alloced_space = 1;
    graph->edges = (edge_t *) malloc(sizeof(edge_t));

    FILE* dataset = fopen(dataset_path, "r");

    if(dataset == NULL) {
        // Couldn't open dataset
        fputs("Couldn't open dataset file!\n", stderr);
        free(graph->edges);
        free(graph);
        return NULL;
    }

    char line[BUF_SIZE];
    // Read dataset line by line and parse the result
    fputs("Reading input graph...\n", stderr);
    unsigned long long linenr = 0;
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

            int returned = graph_edge_add_connection(graph, from + is_zero_indexed, to + is_zero_indexed);
            if(returned != EXIT_SUCCESS) {
                fputs("Couldn't add edge\n", stderr);
                free_graph_edge(graph);
                fclose(dataset);
            }
        } else {
            fprintf(stderr, "Discarding invalid line at %llu\n", linenr);
        }
    } while(line != NULL);

    shrink_graph_edge_to_size(graph);

    fprintf(stderr, "Graph contains %" PRIu32 " nodes and %" PRIedgecount " edges\n", graph->node_count, graph->edge_count);

    fclose(dataset);
    return graph;
}

static int output_bfs_results(result_t* results, uint32_t num_nodes, const char* kind) {
    char output_path[BUF_SIZE];

    if(results_commandline_argument == NULL) {
        sprintf(output_path, "%s-bfsresult.output", kind);
    } else {
        sprintf(output_path, "%s", results_commandline_argument);
    }

    FILE* ofile = fopen(output_path, "w");

    if(ofile != NULL) {
        uint32_t max_depth = 0;

        for(uint32_t node_id = 0; node_id < num_nodes; node_id++) {
            const char* state_str = NULL;

            switch(results[node_id].state) {
                case node_unvisited:
                    state_str = "UNREACHABLE";
                    break;
                case node_reachable:
                    state_str = "REACHABLE";
                    break;
                case node_visited:
                    state_str = "VISITED";
                    break;
                case node_toprocess:
                    state_str = "TOUCHED(INVALID)"; // Should never happen
                    break;
            }
            max_depth = MAX(max_depth, results[node_id].depth);

            fprintf(ofile, "Node %" PRIu32 " - state %s - depth %" PRIu32 "\n", node_id, state_str, results[node_id].depth);
        }

        fclose(ofile);

        fprintf(stderr, "Max graph depth: %" PRIu32 "\nResults written to \"%s\"\n", max_depth, output_path);

        return EXIT_SUCCESS;
    } else {
        uint32_t max_depth = 0;

        for(uint32_t node_id = 0; node_id < num_nodes; node_id++) {
            max_depth = MAX(max_depth, results[node_id].depth);
        }

        fprintf(stderr, "Max graph depth: %" PRIu32 "\nCouldn't write to output file \"%s\"\n",max_depth, output_path);

        return EXIT_FAILURE;
    }
}

#pragma GCC diagnostic pop

#ifdef __cplusplus
}
#endif

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
