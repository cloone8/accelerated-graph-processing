#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "pagerank_omp.h"
#include "../utils/pagerank_graph_utils.h"
#include "../highrestimer.h"

pagerank_t* do_pagerank_omp_dynamic(graph_acc_t* graph, int testrun, double* runtime) {
    fputs("Starting host -> device memory transfers\n", stderr);

    highrestimer_t start_time_no_mem, end_time_no_mem;


    pagerank_t* ranks = init_pagerank(graph->node_count);
    pagerank_t* ranks_next = init_pagerank(graph->node_count);

    float max_threshold;
    float* thresholds = malloc(graph->node_count * sizeof(float));
    for(uint32_t i = 0; i < graph->node_count; i++) {
        thresholds[i] = 0.0f;
    }

    fputs("Starting computation\n", stderr);

    start_time_no_mem = get_highrestime();
    int num_same = 0;
    float prev_threshold = 0;
    do {
        // Do pagerank itself
        #pragma omp parallel for schedule(dynamic)
        for(uint32_t i = 0; i < graph->node_count; i++) {
            pagerank_t rank = 0;

            for(uint32_t j = 0; j < graph->nodes[i].in_count; j++) {
                uint32_t in_index = graph->nodes[i].in[j] - 1;
                rank += (ranks[in_index] / graph->nodes[in_index].out_count);
            }

            ranks_next[i] = ((1.0f - PAGERANK_D) / graph->node_count) + (PAGERANK_D * rank);

            // Determine the threshold
            thresholds[i] = (fabsf(ranks[i] - ranks_next[i]) / ranks[i]) * 100;
        }

        // Get the maximum threshold and shift the ranks
        max_threshold = 0.0f;
        #pragma omp parallel for reduction(max:max_threshold)
        for(uint32_t i = 0; i < graph->node_count; i++) {
            max_threshold = fmaxf(max_threshold, thresholds[i]);
            ranks[i] = ranks_next[i];
        }

        // Pagerank sometimes has some weird precision issues which stops the program from
        // terminating. We solve this by stopping pagerank when the thresholds dont change
        if((fabsf(max_threshold - prev_threshold) < 0.0001f)) {
            if(++num_same > 10) {
                break;
            }
        } else {
            num_same = 0;
            prev_threshold = max_threshold;
        }
    } while(max_threshold > PAGERANK_THRESHOLD);

    fputs("Pagerank converged\n", stderr);

    end_time_no_mem = get_highrestime();

    free(ranks_next);
    free(thresholds);

    double runtime_no_mem = highrestime_diff(start_time_no_mem, end_time_no_mem);

    if(!testrun) {
        printf("%f %f %f\n", runtime_no_mem, runtime_no_mem, 0.0f);
    } else {
        fprintf(stderr, "%f %f %f\n", runtime_no_mem, runtime_no_mem, 0.0f);
        *runtime = runtime_no_mem;
    }

    return ranks;
}

pagerank_t* do_pagerank_omp_static(graph_acc_t* graph, int testrun, double* runtime) {
    fputs("Starting host -> device memory transfers\n", stderr);

    highrestimer_t start_time_no_mem, end_time_no_mem;


    pagerank_t* ranks = init_pagerank(graph->node_count);
    pagerank_t* ranks_next = init_pagerank(graph->node_count);

    float max_threshold;
    float* thresholds = malloc(graph->node_count * sizeof(float));
    for(uint32_t i = 0; i < graph->node_count; i++) {
        thresholds[i] = 0.0f;
    }

    fputs("Starting computation\n", stderr);

    start_time_no_mem = get_highrestime();
    int num_same = 0;
    float prev_threshold = 0;
    do {
        // Do pagerank itself
        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < graph->node_count; i++) {
            pagerank_t rank = 0;

            for(uint32_t j = 0; j < graph->nodes[i].in_count; j++) {
                uint32_t in_index = graph->nodes[i].in[j] - 1;
                rank += (ranks[in_index] / graph->nodes[in_index].out_count);
            }

            ranks_next[i] = ((1.0f - PAGERANK_D) / graph->node_count) + (PAGERANK_D * rank);

            // Determine the threshold
            thresholds[i] = (fabsf(ranks[i] - ranks_next[i]) / ranks[i]) * 100;
        }

        // Get the maximum threshold and shift the ranks
        max_threshold = 0.0f;
        #pragma omp parallel for reduction(max:max_threshold)
        for(uint32_t i = 0; i < graph->node_count; i++) {
            max_threshold = fmaxf(max_threshold, thresholds[i]);
            ranks[i] = ranks_next[i];
        }

        // Pagerank sometimes has some weird precision issues which stops the program from
        // terminating. We solve this by stopping pagerank when the thresholds dont change
        if((fabsf(max_threshold - prev_threshold) < 0.0001f)) {
            if(++num_same > 10) {
                break;
            }
        } else {
            num_same = 0;
            prev_threshold = max_threshold;
        }
    } while(max_threshold > PAGERANK_THRESHOLD);

    fputs("Pagerank converged\n", stderr);

    end_time_no_mem = get_highrestime();

    free(ranks_next);
    free(thresholds);

    double runtime_no_mem = highrestime_diff(start_time_no_mem, end_time_no_mem);

    if(!testrun) {
        printf("%f %f %f\n", runtime_no_mem, runtime_no_mem, 0.0f);
    } else {
        fprintf(stderr, "%f %f %f\n", runtime_no_mem, runtime_no_mem, 0.0f);
        *runtime = runtime_no_mem;
    }

    return ranks;
}
