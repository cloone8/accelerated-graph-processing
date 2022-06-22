#include <stdlib.h>
#include <stdio.h>

#include "main.h"
#include "utils/bfs_graph_utils.h"
#include "bfs/bfs_cu.h"

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

    graph_edge_t* graph = parse_input_file_edge(dataset_path);

    if(graph == NULL) {
        fputs("Error reading dataset file!\n", stderr);
        return EXIT_FAILURE;
    }

    fputs("Finished initialisation\n", stderr);

    result_t* depths = NULL;

    for(int i = 0; i < (NUM_EXPERIMENTS * 32); i++) {
        depths = do_bfs_cuda(graph);
        if(depths == NULL) {
            fputs("Something went wrong during graph processing!\n", stderr);
            free_graph_edge(graph);
            return EXIT_FAILURE;
        }

        if((i + 1) < (NUM_EXPERIMENTS * 32)) {
            free(depths);
        }
    }

    output_bfs_results(depths, graph->node_count, "cuda");

    free_graph_edge(graph);
    free(depths);

    return EXIT_SUCCESS;
}
