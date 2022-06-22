#ifndef __PAGERANK_ACC_UTILS_C__
#define __PAGERANK_ACC_UTILS_C__

#include "pagerank_graph_utils.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <limits.h>
#include <inttypes.h>

#include "../main.h"

// Structs for the OpenACC implementation

typedef struct node_acc {
    uint32_t out_count;
    uint32_t __out_alloced;
    uint32_t* out;

    uint32_t in_count;
    uint32_t __in_alloced;
    uint32_t* in;
} node_acc_t;

typedef struct graph_acc {
    uint32_t node_count;
    uint32_t __alloced_space;
    node_acc_t* nodes;
} graph_acc_t;

static int realloc_graph_node_array_acc(graph_acc_t* graph, unsigned int new_size) {
    uint32_t starting_node_count = graph->node_count;

    if(new_size > graph->__alloced_space) {
        uint32_t new_alloced_space = MAX((1.3333 * graph->__alloced_space), graph->__alloced_space + 10);
        new_alloced_space = MAX(new_alloced_space, new_size + 10);

        graph->nodes = (node_acc_t*) realloc(graph->nodes, new_alloced_space * sizeof(node_acc_t));

        if(graph->nodes == NULL) {
            fputs("Memory allocation failure\n", stderr);
            return 0;
        }

        graph->__alloced_space = new_alloced_space;
    }

    // Modify the node_count after the realloc error check because
    // realloc only changes the nodes array size when there is no error
    graph->node_count = new_size;

    for(uint32_t i = starting_node_count; i < new_size; i++) {
        graph->nodes[i].in_count = 0;
        graph->nodes[i].__in_alloced = 1;
        graph->nodes[i].in = (uint32_t*) malloc(sizeof(uint32_t));

        graph->nodes[i].out = (uint32_t*) malloc(sizeof(uint32_t));
        graph->nodes[i].__out_alloced = 1;
        graph->nodes[i].out_count = 0;
    }

    return 1;
}

static int node_add_connection_acc(graph_acc_t* graph, unsigned int from, unsigned int to) {
    void* realloc_result;

    graph->nodes[from - 1].out_count++;
    graph->nodes[to - 1].in_count++;

    // Check if there is enough space for the new outgoing connection
    if(graph->nodes[from - 1].__out_alloced < graph->nodes[from - 1].out_count) {
        uint32_t new_alloc_size = MAX((1.3333 * graph->nodes[from - 1].__out_alloced), graph->nodes[from - 1].__out_alloced + 10);
        graph->nodes[from - 1].__out_alloced = new_alloc_size;

        realloc_result = realloc(graph->nodes[from - 1].out, graph->nodes[from - 1].__out_alloced * sizeof(uint32_t));

        if(realloc_result == NULL) {
            fputs("Memory allocation failure\n", stderr);
            return 0;
        }

        graph->nodes[from - 1].out = (uint32_t*) realloc_result;
    }

    // Check if there is enough space for the new incoming connection
    if(graph->nodes[to - 1].__in_alloced < graph->nodes[to - 1].in_count) {
        uint32_t new_alloc_size = MAX((1.3333 * graph->nodes[to - 1].__in_alloced), graph->nodes[to - 1].__in_alloced + 10);
        graph->nodes[to - 1].__in_alloced = new_alloc_size;

        realloc_result = realloc(graph->nodes[to - 1].in, graph->nodes[to - 1].__in_alloced * sizeof(uint32_t));

        if(realloc_result == NULL) {
            fputs("Memory allocation failure\n", stderr);
            return 0;
        }
        graph->nodes[to - 1].in = (uint32_t*) realloc_result;
    }

    // Add the connections
    graph->nodes[from - 1].out[graph->nodes[from - 1].out_count - 1] = to;
    graph->nodes[to - 1].in[graph->nodes[to - 1].in_count - 1] = from;

    return 1;
}

static int graph_handle_sinks_acc(graph_acc_t* graph) {
    fputs("Finding and handling sinks\n", stderr);

    for(uint32_t i = 0; i < graph->node_count; i++) {
        if(graph->nodes[i].out_count == 0) {
            for(uint32_t j = 0; j < graph->nodes[i].in_count; j++) {
                if(!node_add_connection_acc(graph, i + 1, graph->nodes[i].in[j])) {
                    return EXIT_FAILURE;
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

static void shrink_graph_acc_to_size(graph_acc_t* graph) {
    fputs("Shrinking graph to proper memory size\n", stderr);

    graph->nodes = (node_acc_t*) realloc(graph->nodes, graph->node_count * sizeof(node_acc_t));
    graph->__alloced_space = graph->node_count;

    for(uint32_t i = 0; i < graph->node_count; i++) {
        node_acc_t* node = &graph->nodes[i];
        node->in = (uint32_t*) realloc(node->in, node->in_count * sizeof(uint32_t));
        node->__in_alloced = node->in_count;
        node->out = (uint32_t*) realloc(node->out, node->out_count * sizeof(uint32_t));
        node->__out_alloced = node->out_count;
    }
}

static void free_graph_pagerank_acc(graph_acc_t* tofree) {
    for(uint32_t i = 0; i < tofree->node_count; i++) {
        free(tofree->nodes[i].in);
        free(tofree->nodes[i].out);
    }

    free(tofree->nodes);
    free(tofree);
}

static graph_acc_t* parse_input_file_pagerank_acc(char* dataset_path) {
    // Initialise the graph structure
    graph_acc_t* graph = (graph_acc_t*) malloc(sizeof(graph_acc_t));

    graph->node_count = 0;
    graph->__alloced_space = 1;
    graph->nodes = (node_acc_t*) malloc(sizeof(node_acc_t));

    FILE* dataset = fopen(dataset_path, "r");

    if(dataset == NULL) {
        // Couldn't open dataset
        fputs("Couldn't open dataset file!\n", stderr);
        free(graph->nodes);
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
                if(!realloc_graph_node_array_acc(graph, MAX(from, to))) {
                    fputs("Couldn't acquire memory to expand graph node array\n", stderr);
                    free_graph_pagerank_acc(graph);
                    fclose(dataset);
                    return NULL;
                }
            }

            if(!node_add_connection_acc(graph, from, to)) {
                fputs("Couldn't add node connection\n", stderr);
                free_graph_pagerank_acc(graph);
                fclose(dataset);
                return NULL;
            }

            edgecount++;
        } else {
            fprintf(stderr, "Discarding invalid line at %llu\n", linenr);
        }
    } while(line != NULL);

    if(graph_handle_sinks_acc(graph) != EXIT_SUCCESS) {
        fputs("Couldn't add node connection\n", stderr);
        free_graph_pagerank_acc(graph);
        fclose(dataset);
        return NULL;
    }
    shrink_graph_acc_to_size(graph);

    fprintf(stderr, "Graph contains %" PRIu32 " nodes and %llu edges\n", graph->node_count, edgecount);

    fclose(dataset);
    return graph;
}

#endif
