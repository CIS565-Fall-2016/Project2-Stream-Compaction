#include "naive.h"

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace StreamCompaction {
namespace Naive {

    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

// DONE: __global__
__global__ void kernNaiveScanPass(int N, int offset, int* in_buffer, int* out_buffer)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    if (index >= offset)
    {
        out_buffer[index] = in_buffer[index - offset] + in_buffer[index];
    }
    else
    {
        out_buffer[index] = in_buffer[index];
    }
}

int getNaiveScanBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernNaiveScanPass);
    }
    return block_size;
}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) 
{
    if (n <= 0) { return; }

    auto block_size = getNaiveScanBlockSize();
    auto full_blocks_per_grid = (n + block_size - 1) / block_size;

    // DONE
    int* dev_in_buffer;
    cudaMalloc((void**)&dev_in_buffer, n * sizeof(*dev_in_buffer));
    checkCUDAError("cudaMalloc dev_in_buffer failed!");
    int* dev_out_buffer;
    cudaMalloc((void**)&dev_out_buffer, n * sizeof(*dev_out_buffer));
    checkCUDAError("cudaMalloc dev_out_buffer failed!");

    cudaMemcpy(dev_in_buffer, idata, n * sizeof(*idata), cudaMemcpyHostToDevice);
    
    timer().startGpuTimer();

    auto cap = ilog2ceil(n);
    int offset;
    for (int d = 1; d <= cap; d++)
    {
        offset = 1 << (d - 1);
        kernNaiveScanPass <<< full_blocks_per_grid, block_size >>>(n, offset, dev_in_buffer, dev_out_buffer);
        std::swap(dev_in_buffer, dev_out_buffer);
    }
    std::swap(dev_in_buffer, dev_out_buffer);

    timer().endGpuTimer();

    // defered copy because of exclusive scan
    cudaMemcpy(odata + 1, dev_out_buffer, (n - 1) * sizeof(*odata), cudaMemcpyDeviceToHost);
    odata[0] = 0;

    cudaFree(dev_in_buffer);
    cudaFree(dev_out_buffer);
}

}
}
