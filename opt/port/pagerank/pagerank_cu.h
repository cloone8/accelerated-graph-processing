#ifndef __PAGERANK_CU_H__
#define __PAGERANK_CU_H__

#include "../utils/pagerank_graph_utils.h"

#define THREAD_BLOCK_COUNT (256)

pagerank_t* do_pagerank_cu(graph_cuda_t* graph);

#endif
