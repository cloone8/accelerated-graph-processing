#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "pagerank_acc.h"
#include "../utils/pagerank_graph_utils.h"
#include "../highrestimer.h"

static void copy_graph_to_gpu(graph_acc_t* graph) {
    #pragma acc enter data copyin(graph[0:1]) async
    #pragma acc enter data copyin(graph->nodes[0:graph->node_count]) async

    for(uint32_t i = 0; i < graph->node_count; i++) {
        #pragma acc enter data \
            copyin(graph->nodes[i].in[0:graph->nodes[i].in_count]) async
    }
}

static void free_graph_from_gpu(graph_acc_t* graph) {
    for(uint32_t i = 0; i < graph->node_count; i++) {
        #pragma acc exit data delete(graph->nodes[i].in[0:graph->nodes[i].in_count]) async
    }

    #pragma acc exit data delete(graph->nodes[0:graph->node_count]) async
    #pragma acc exit data delete(graph[0:1]) async
}

pagerank_t* do_pagerank_acc(graph_acc_t* graph) {
    fputs("Starting host -> device memory transfers\n", stderr);

    highrestimer_t start_time, end_time, start_time_no_mem, end_time_no_mem;

    start_time = get_highrestime();

    pagerank_t* ranks = init_pagerank(graph->node_count);
    pagerank_t* ranks_next = init_pagerank(graph->node_count);

    float max_threshold;
    float* thresholds = malloc(graph->node_count * sizeof(float));
    for(uint32_t i = 0; i < graph->node_count; i++) {
        thresholds[i] = 0.0f;
    }

    // Copy the graph to the GPU
    copy_graph_to_gpu(graph);
    #pragma acc enter data copyin(ranks[0:graph->node_count]) async
    #pragma acc enter data copyin(ranks_next[0:graph->node_count]) async
    #pragma acc enter data copyin(thresholds[0:graph->node_count]) async

    #pragma acc wait

    fputs("Memory transfer done, starting computation\n", stderr);

    #pragma acc data present(ranks[0:graph->node_count], ranks_next[0:graph->node_count])
    #pragma acc data present(thresholds[0:graph->node_count])
    #pragma acc data present(graph->nodes->in, graph->nodes, graph)
    {
        start_time_no_mem = get_highrestime();
        int num_same = 0;
        float prev_threshold = 0;
        do {
            // Do pagerank itself
            #pragma acc parallel loop async vector_length(32)
            for(uint32_t i = 0; i < graph->node_count; i++) {
                pagerank_t rank = 0;

                #pragma acc loop vector reduction(+:rank)
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
            #pragma acc parallel loop reduction(max:max_threshold) async
            for(uint32_t i = 0; i < graph->node_count; i++) {
                max_threshold = fmaxf(max_threshold, thresholds[i]);
                ranks[i] = ranks_next[i];
            }

            #pragma acc wait

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

        fputs("Pagerank converged, freeing GPU memory\n", stderr);

        end_time_no_mem = get_highrestime();

        #pragma acc update host(ranks[0:graph->node_count]) async
    }

    // Free GPU memory
    #pragma acc exit data delete(ranks_next[0:graph->node_count]) async
    #pragma acc exit data delete(ranks[0:graph->node_count]) async
    #pragma acc exit data delete(thresholds[0:graph->node_count]) async
    free_graph_from_gpu(graph);

    #pragma acc wait
    free(ranks_next);
    free(thresholds);

    end_time = get_highrestime();

    double runtime = highrestime_diff(start_time, end_time);
    double runtime_no_mem = highrestime_diff(start_time_no_mem, end_time_no_mem);

    printf("%f %f %f\n", runtime, runtime_no_mem, runtime - runtime_no_mem);

    return ranks;
}
