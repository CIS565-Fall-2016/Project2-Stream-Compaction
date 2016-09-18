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

inline int fullBlocksPerGrid(int n, int block_size)
{
    return (n + block_size - 1) / block_size;
}

template<typename T>
int calculateBlockSizeForDeviceFunction(T func)
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func);
    return block_size;
}


namespace StreamCompaction {
namespace Common {
    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    //__global__ void kernScatter(int n, int *odata,
    //        const int *idata, const int *bools, const int *indices);
    __global__ void kernScatter(int n, int *odata,
            const int *idata, const int *indices); // use one less buffer to save space

    int getMapToBooleanBlockSize();
    int getScatterBlocksize();
}
}

