#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}


namespace StreamCompaction {
namespace Common {
    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    __global__ void kernScatter(int n, int *odata,
            const int *idata, const int *bools, const int *indices);
}
}

#define START_CUDA_TIMER() \
  cudaEvent_t start, stop; \
  cudaEventCreate(&start); \
  cudaEventCreate(&stop); \
  cudaEventRecord(start);

#define STOP_CUDA_TIMER() \
  cudaEventRecord(stop); \
  cudaEventSynchronize(stop); \
  float milliseconds = 0; \
  cudaEventElapsedTime(&milliseconds, start, stop); \
  cudaEventDestroy(start); \
  cudaEventDestroy(stop); \
  printf("Elapsed: %fms\n", milliseconds);
