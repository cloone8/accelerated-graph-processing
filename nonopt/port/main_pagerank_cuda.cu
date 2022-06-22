#include <stdlib.h>
#include <stdio.h>

#include "main.h"
#include "utils/pagerank_graph_utils.h"
#include "pagerank/pagerank_cu.h"

char* results_commandline_argument = NULL;

static inline void print_usage(char* executable_name) {
    printf("Usage: %s [path_to_dataset]\n", executable_name);
    printf("   or: %s [path_to_dataset] [path_to_results]\n", executable_name);
}

int main(int argc, char* argv[]) {
    char* dataset_path = NULL;

    if(argc < 2) {
        fputs("Invalid number of arguments\n", stderr);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    } else if(argc == 2) {
        dataset_path = argv[1];
    } else if(argc == 3) {
        dataset_path = argv[1];
        results_commandline_argument = argv[2];
    } else {
        fputs("Too many input arguments!\n", stderr);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    graph_cuda_t* graph = parse_input_file_pagerank_cuda(dataset_path);

    if(graph == NULL) {
        fputs("Error reading dataset file!\n", stderr);
        return EXIT_FAILURE;
    }

    fputs("Finished initialisation\n", stderr);

    pagerank_t* ranks = NULL;

    for(int i = 0; i < NUM_EXPERIMENTS; i++) {
        ranks = do_pagerank_cu(graph);

        if(ranks == NULL) {
            fputs("Something went wrong during graph processing!\n", stderr);
            free_graph_pagerank_cuda(graph);
            return EXIT_FAILURE;
        }

        // Sanity check
        long double sum = 0.0;
        for(uint32_t j = 0; j < graph->node_count; j++) {
            sum += ranks[j];
        }
        fprintf(stderr, "Pagerank sum: %Lf\n", sum);

        if((i + 1) < NUM_EXPERIMENTS) {
            free_pagerank(ranks);
        }
    }

    print_results(ranks, graph->node_count, "cuda");

    free_graph_pagerank_cuda(graph);
    free_pagerank(ranks);

    return EXIT_SUCCESS;
}
