#include "radix_sort.h"

#include "efficient.h"
#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace StreamCompaction {
namespace RadixSort {


using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}


__global__ void kernComputeBArray(int N, int bit_mask, bool *b, const int *idata)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    b[index] = ((idata[index] & bit_mask) != 0);
}

__global__ void kernComputeEArray(int N, int* e, const bool *b)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    // e is int array since directly performing scan on it
    e[index] = !b[index];
}

__global__ void kernComputeDArray(int N, int* d, const int* f, const bool* b)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    // t[i] = i-f[i] + total_falses
    // total_falses = e[n-1] + f[n-1] = !b[n-1] + f[n-1]
    // d[i] = b[i]? t[i] : f[i]
    if (b[index])
    {
        d[index] = index - f[index] + !b[N - 1] + f[N - 1];
    }
    else
    {
        d[index] = f[index];
    }
}

__global__ void kernReshuffle(int N, int* to_buffer, const int* from_buffer, const int* indices)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    to_buffer[indices[index]] = from_buffer[index];
}

int getComputeBArrayBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernComputeBArray);
    }
    return block_size;
}

int getComputeEArrayBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernComputeEArray);
    }
    return block_size;
}

int getComputeDArrayBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernComputeDArray);
    }
    return block_size;
}

int getReshuffleBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernReshuffle);
    }
    return block_size;
}

/**
* performs radix sort, which assumes all data is int type and between [0, max_value)
*/
void radixSort(int* start, int* end, int max_value)
{
    auto n = static_cast<std::size_t>(end - start);

    auto block_size_compute_b = getComputeBArrayBlockSize();
    auto full_blocks_per_grid_compute_b = fullBlocksPerGrid(n, block_size_compute_b);

    auto block_size_compute_e = getComputeBArrayBlockSize();
    auto full_blocks_per_grid_compute_e = fullBlocksPerGrid(n, block_size_compute_e);

    auto block_size_compute_d = getComputeBArrayBlockSize();
    auto full_blocks_per_grid_compute_d = fullBlocksPerGrid(n, block_size_compute_d);

    auto block_size_reshuffle = getReshuffleBlockSize();
    auto full_blocks_per_grid_reshuffle = fullBlocksPerGrid(n, block_size_compute_d);

    auto extended_n = std::size_t(1) << ilog2ceil(n); // round up to power of two for scanning

    int* dev_array;
    cudaMalloc((void**)&dev_array, n * sizeof(*dev_array));
    checkCUDAError("cudaMalloc dev_array failed!");
    cudaMemcpy(dev_array, start, n * sizeof(*start), cudaMemcpyHostToDevice);

    int* dev_temp;
    cudaMalloc((void**)&dev_temp, n * sizeof(*dev_temp));
    checkCUDAError("cudaMalloc dev_temp failed!");

    bool* dev_b; // buffer which holds values of b
    cudaMalloc((void**)&dev_b, n * sizeof(*dev_b));
    checkCUDAError("cudaMalloc dev_b failed!");

    int* dev_ef; // buffer which holds values of e and f
    cudaMalloc((void**)&dev_ef, extended_n * sizeof(*dev_ef));
    checkCUDAError("cudaMalloc dev_ef failed!");

    int* dev_d; // buffer which holds values of d
    cudaMalloc((void**)&dev_d, n * sizeof(*dev_d));
    checkCUDAError("cudaMalloc dev_d failed!");

    timer().startGpuTimer();
    // input betweem [0, max_value)
    // auto lsb_offset = 0;
    auto msb_offset = ilog2ceil(max_value);
    for (int offset = 0; offset < msb_offset; offset++)
    {
        auto bit_mask = 1 << offset;
        kernComputeBArray <<<full_blocks_per_grid_compute_b, block_size_compute_b >>>(n, bit_mask, dev_b, dev_array);
        kernComputeEArray <<<full_blocks_per_grid_compute_e, block_size_compute_e >>>(n, dev_ef, dev_b);

        StreamCompaction::Efficient::scanInPlaceDevice(extended_n, dev_ef);

        kernComputeDArray <<<full_blocks_per_grid_compute_d, block_size_compute_d >>>(n, dev_d, dev_ef, dev_b);

        kernReshuffle <<<full_blocks_per_grid_reshuffle, block_size_reshuffle >>>(n, dev_temp, dev_array, dev_d);
        std::swap(dev_temp, dev_array);
        
    }
    timer().endGpuTimer();

    cudaMemcpy(start, dev_array, n * sizeof(*start), cudaMemcpyDeviceToHost);

    cudaFree(dev_array);
    cudaFree(dev_temp);
    cudaFree(dev_b);
    cudaFree(dev_ef);
    cudaFree(dev_d);
}

}
}
