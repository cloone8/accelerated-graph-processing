#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <time.h>

#include "bfs_acc.h"
#include "../utils/bfs_graph_utils.h"
#include "../highrestimer.h"

result_t* do_bfs_acc(graph_edge_t* graph) {
    fputs("Starting OpenACC BFS\n", stderr);

    int was_updated;
    highrestimer_t start_time, end_time, start_time_no_mem, end_time_no_mem;
    const edge_t* edges = graph->edges;
    const edge_count_t edge_count = graph->edge_count;
    const uint32_t node_count = graph->node_count;
    result_t* results = (result_t *) malloc(node_count * sizeof(result_t));

    for(uint32_t i = 0; i < graph->node_count; i++) {
        results[i].state = node_unvisited;
        results[i].depth = 0;
    }

    results[0].state = node_toprocess;

    start_time = get_highrestime();

    // Copy data to GPU
    fputs("Copying data to GPU\n", stderr);


    #pragma acc data copy(results[0:node_count]) copyin(edges[0:edge_count])
    {
        // Do BFS
        fputs("Done copying, starting BFS\n", stderr);
        start_time_no_mem = get_highrestime();
        do {
            was_updated = 0;

            // Do BFS search
            #pragma acc parallel loop gang vector async
            for(edge_count_t i = 0; i < edge_count; i++) {
                uint32_t origin_index = edges[i].from - 1;
                uint32_t destination_index = edges[i].to - 1;

                if(results[origin_index].state == node_toprocess) {
                    if(results[destination_index].state == node_unvisited) {
                        results[destination_index].state = node_reachable;
                        results[destination_index].depth = results[origin_index].depth + 1;
                    }
                }
            }

            // Update states
            #pragma acc parallel loop gang vector async
            for(uint32_t i = 0; i < node_count; i++) {
                switch(results[i].state) {
                    case node_unvisited:
                    case node_visited:
                        break;
                    case node_reachable:
                        results[i].state = node_toprocess;
                        was_updated = 1;
                        break;
                    case node_toprocess:
                        results[i].state = node_visited;
                        was_updated = 1;
                        break;
                }
            }

            #pragma acc wait
        } while(was_updated == 1);
        end_time_no_mem = get_highrestime();
        fputs("BFS done, copying results back and freeing GPU memory\n", stderr);
    }

    end_time = get_highrestime();
    fputs("Done copying results and freeing GPU memory\n", stderr);

    double runtime = highrestime_diff(start_time, end_time);
    double runtime_no_mem = highrestime_diff(start_time_no_mem, end_time_no_mem);

    printf("%f %f %f\n", runtime, runtime_no_mem, runtime - runtime_no_mem);

    return results;
}
