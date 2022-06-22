# CC=gcc-8
CC=pgcc
NVCC=nvcc
NVCCFLAGS=-O3

ifeq ($(CC), pgcc)
CFLAGS=-acc -mp -fast -Minfo $(CFLAGS_EXTRA)
LDFLAGS=
LIBS=
else
CFLAGS=-Wall -Wextra -std=c99 -pedantic -fopenmp -fopenacc -O3
LDFLAGS=
LIBS=
endif

BFSACCFILES=bfs/bfs_acc.o
BFSCUDAFILES=bfs/bfs_cu.o
PAGERANKACCFILES=pagerank/pagerank_acc.o
PAGERANKOMPFILES=pagerank/pagerank_omp.o
PAGERANKCUDAFILES=pagerank/pagerank_cu.o

EXEC_NAME_BFS_ACC=tester_bfs_acc
EXEC_NAME_BFS_CUDA=tester_bfs_cuda
EXEC_NAME_PAGERANK_ACC=tester_pagerank_acc
EXEC_NAME_PAGERANK_ACC_CPU=tester_pagerank_acccpu
EXEC_NAME_PAGERANK_OMP=tester_pagerank_omp
EXEC_NAME_PAGERANK_CUDA=tester_pagerank_cuda

.PHONY: clean

all: pagerank bfs

cpu: clean_cpu pagerank_acc_cpu pagerank_omp

bfs: bfs_acc bfs_cuda

bfs_acc: $(BFSACCFILES) main_bfs_acc.o
	@echo "Linking $(EXEC_NAME_BFS_ACC)"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXEC_NAME_BFS_ACC) $^ $(LIBS)

bfs_cuda: $(BFSCUDAFILES) main_bfs_cuda.o
	@echo "Linking $(EXEC_NAME_BFS_CUDA)"
	@$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $(EXEC_NAME_BFS_CUDA) $^ $(LIBS)

pagerank: pagerank_acc pagerank_cuda

pagerank_omp: $(PAGERANKOMPFILES) main_pagerank_omp.o
	@echo "Linking $(EXEC_NAME_PAGERANK_OMP)"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXEC_NAME_PAGERANK_OMP) $^ $(LIBS)

pagerank_acc: $(PAGERANKACCFILES) main_pagerank_acc.o
	@echo "Linking $(EXEC_NAME_PAGERANK_ACC)"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXEC_NAME_PAGERANK_ACC) $^ $(LIBS)

pagerank_acc_cpu: $(PAGERANKACCFILES) main_pagerank_acc.o
	@echo "Linking $(EXEC_NAME_PAGERANK_ACC_CPU)"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXEC_NAME_PAGERANK_ACC_CPU) $^ $(LIBS)

pagerank_cuda: $(PAGERANKCUDAFILES) main_pagerank_cuda.o
	@echo "Linking $(EXEC_NAME_PAGERANK_CUDA)"
	@$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $(EXEC_NAME_PAGERANK_CUDA) $^ $(LIBS)

%.o: %.c
	@echo "Compiling file $<"
	@$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	@echo "Compiling cuda-enabled file $<"
	@$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean_cpu:
	@echo "Cleaning"
	@rm -f $(BFSACCFILES)
	@rm -f $(BFSCUDAFILES)
	@rm -f $(PAGERANKACCFILES)
	@rm -f $(PAGERANKOMPFILES)
	@rm -f $(PAGERANKCUDAFILES)
	@rm -f main_bfs_acc.o
	@rm -f main_bfs_cuda.o
	@rm -f main_pagerank_acc.o
	@rm -f main_pagerank_omp.o
	@rm -f main_pagerank_cuda.o
	@rm -f *.output

clean:
	@echo "Cleaning"
	@rm -f $(BFSACCFILES)
	@rm -f $(BFSCUDAFILES)
	@rm -f $(PAGERANKACCFILES)
	@rm -f $(PAGERANKOMPFILES)
	@rm -f $(PAGERANKCUDAFILES)
	@rm -f main_bfs_acc.o
	@rm -f main_bfs_cuda.o
	@rm -f main_pagerank_acc.o
	@rm -f main_pagerank_omp.o
	@rm -f main_pagerank_cuda.o
	@rm -f $(EXEC_NAME_BFS_ACC)
	@rm -f $(EXEC_NAME_BFS_CUDA)
	@rm -f $(EXEC_NAME_PAGERANK_ACC)
	@rm -f $(EXEC_NAME_PAGERANK_ACC_CPU)
	@rm -f $(EXEC_NAME_PAGERANK_CUDA)
	@rm -f $(EXEC_NAME_PAGERANK_OMP)
	@rm -f *.output
