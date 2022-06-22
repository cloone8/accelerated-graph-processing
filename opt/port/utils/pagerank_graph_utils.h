#ifndef __PAGERANK_GRAPH_UTILS_H__
#define __PAGERANK_GRAPH_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

/**
 *  This file has a header-only implementation of the PAGERANK graph utilities
 *  and structure, to circumvent weird cross-compilation issues
 */

typedef float pagerank_t;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define PAGERANK_D (0.85f)
#define PAGERANK_THRESHOLD (0.001f)

#include "pagerank_acc_utils.c"
#include "pagerank_cuda_utils.c"

static pagerank_t* init_pagerank(uint32_t nodecount) {
    pagerank_t* ranks = (pagerank_t*) malloc(nodecount * sizeof(pagerank_t));

    for(uint32_t i = 0; i < nodecount; i++) {
        ranks[i] = 1.0f / nodecount;
    }

    return ranks;
}

static void free_pagerank(pagerank_t* ranks) {
    free(ranks);
}


static void print_results(pagerank_t* ranks, uint32_t nodecount, const char* kind) {
    char output_path[BUF_SIZE];

    if(results_commandline_argument == NULL) {
        sprintf(output_path, "%sresult.output", kind);
    } else {
        sprintf(output_path, "%s", results_commandline_argument);
    }

    FILE* ofile = fopen(output_path, "w");

    if(ofile != NULL) {
        for(uint32_t node_id = 0; node_id < nodecount; node_id++) {
            fprintf(ofile, "Node %" PRIu32 " - Pagerank %.20f\n", node_id + 1, ranks[node_id]);
        }

        fclose(ofile);
    } else {
        fprintf(stderr, "Couldn't write to output file \"%s\"\n", output_path);
    }
}

#pragma GCC diagnostic pop

#ifdef __cplusplus
}
#endif

#endif
