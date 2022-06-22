#include <stdlib.h>
#include <stdio.h>

#include "main.h"
#include "utils/pagerank_graph_utils.h"
#include "pagerank/pagerank_omp.h"

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

    graph_acc_t* graph = parse_input_file_pagerank_acc(dataset_path);

    if(graph == NULL) {
        fputs("Error reading dataset file!\n", stderr);
        return EXIT_FAILURE;
    }

    fputs("Finished initialisation\n", stderr);

    fputs("Testing scheduling algorithms\n", stderr);

    double avg_runtime_dynamic = 0.0;
    double avg_runtime_static = 0.0;

    fputs("Testing dynamic scheduling\n", stderr);
    for(int i = 0; i < 7; i++) {
        double this_runtime = 0.0;
        pagerank_t* ranks = NULL;

        ranks = do_pagerank_omp_dynamic(graph, 1, &this_runtime);
        free_pagerank(ranks);

        avg_runtime_dynamic += this_runtime;
    }

    avg_runtime_dynamic /= 7;

    fputs("Testing static scheduling\n", stderr);
    for(int i = 0; i < 7; i++) {
        double this_runtime = 0.0;
        pagerank_t* ranks = NULL;

        ranks = do_pagerank_omp_static(graph, 1, &this_runtime);
        free_pagerank(ranks);

        avg_runtime_static += this_runtime;
    }

    avg_runtime_static /= 7;

    if(avg_runtime_static < avg_runtime_dynamic) {
        fputs("Picking static scheduling\n", stderr);
    } else {
        fputs("Picking dynamic scheduling\n", stderr);
    }

    pagerank_t* ranks = NULL;

    for(int i = 0; i < NUM_EXPERIMENTS; i++) {

        if(avg_runtime_static < avg_runtime_dynamic) {
            ranks = do_pagerank_omp_static(graph, 0, NULL);
        } else {
            ranks = do_pagerank_omp_dynamic(graph, 0, NULL);
        }

        if(ranks == NULL) {
            fputs("Something went wrong during graph processing!\n", stderr);
            free_graph_pagerank_acc(graph);
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

    print_results(ranks, graph->node_count, "omp");

    free_graph_pagerank_acc(graph);
    free_pagerank(ranks);

    return EXIT_SUCCESS;
}
