#ifndef __PAGERANK_ACC_H__
#define __PAGERANK_ACC_H__

#include "../utils/pagerank_graph_utils.h"

extern pagerank_t* do_pagerank_omp_static(graph_acc_t* graph, int testrun, double* runtime);
extern pagerank_t* do_pagerank_omp_dynamic(graph_acc_t* graph, int testrun, double* runtime);

#endif
