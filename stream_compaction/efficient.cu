#include "efficient.h"

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace Efficient {


// DONE: __global__
__global__ void kernScanUpSweepPass(int N, int add_distance, int* buffer)
{
    // TODO: use less threads?
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    if ((index + 1) % (add_distance * 2) == 0)
    {
        buffer[index] = buffer[index] + buffer[index - add_distance];
    }
}

/**
* Swap value of two array members
*/
__global__ void kernSwap(int index1, int index2, int* buffer)
{
    auto thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index == 0)
    {
        buffer[index1] ^= buffer[index2];
        buffer[index2] ^= buffer[index1];
        buffer[index1] ^= buffer[index2];
    }
}

__global__ void kernScanDownSweepPass(int N, int distance, int* buffer)
{
    // TODO: use less threads?
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    if ((index + 1) % (distance * 2) == 0)
    {
        auto temp = buffer[index - distance];
        buffer[index - distance] = buffer[index];
        buffer[index] = temp + buffer[index];
    }
}


int getUpSweepBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernScanUpSweepPass);
    }
    return block_size;
}

int getDownSweepBlockSize()
{
    // not thread-safe
    static int block_size = -1;
    if (block_size == -1)
    {
        block_size = calculateBlockSizeForDeviceFunction(kernScanDownSweepPass);
    }
    return block_size;
}

enum class ScanType{inclusive, exclusive};

void scan_implemention(int n, int *odata, const int *idata, ScanType scan_type)
{
    if (n <= 0) { return; }

    // DONE
    // round n up to power of two
    auto extended_n = std::size_t(1) << ilog2ceil(n);
    // plus one for 
    auto extended_n_plus_1 = extended_n + 1;

    int* dev_buffer;
    cudaMalloc((void**)&dev_buffer, extended_n_plus_1 * sizeof(*dev_buffer));
    checkCUDAError("cudaMalloc dev_buffer failed!");

    // fill zero and copy to device buffer
    cudaMemset(dev_buffer, 0, extended_n_plus_1 * sizeof(*idata));
    cudaMemcpy(dev_buffer, idata, n * sizeof(*idata), cudaMemcpyHostToDevice);

    auto block_size_up = getUpSweepBlockSize();
    auto full_blocks_per_grid_up = fullBlocksPerGrid(extended_n, block_size_up);
    auto block_size_down = getDownSweepBlockSize();
    auto full_blocks_per_grid_down = fullBlocksPerGrid(extended_n, block_size_down);

    // up sweep
    auto pass_count = ilog2ceil(extended_n) - 1;
    for (int d = 0; d <= pass_count; d++)
    {
        kernScanUpSweepPass << <full_blocks_per_grid_up, block_size_up >> >(extended_n, 1 << d, dev_buffer);
    }

    // swap the last element of up sweep result and the real last element (0)
    kernSwap << <1, 1 >> >(extended_n - 1, extended_n_plus_1 - 1, dev_buffer);

    // down sweep
    for (int d = pass_count; d >= 0; d--)
    {
        kernScanDownSweepPass << <full_blocks_per_grid_down, block_size_down >> >(extended_n, 1 << d, dev_buffer);
    }

    // copy with offset to make it an inclusive scan
    cudaMemcpy(odata, dev_buffer + 1, n * sizeof(*odata), cudaMemcpyDeviceToHost);

    cudaFree(dev_buffer);
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 * This just call scan_implementaion
 */
void scan(int n, int *odata, const int *idata) 
{
    scan_implemention(n, odata, idata, ScanType::inclusive);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    if (n <= 0) { return 0; }
    return -1;

    // TODO

    int* dev_idata;
    cudaMalloc((void**)&dev_idata, n * sizeof(*dev_idata));
    checkCUDAError("cudaMalloc dev_idata failed!");
    cudaMemcpy(dev_idata, idata, n * sizeof(*idata), cudaMemcpyHostToDevice);

    auto extended_n = std::size_t(1) << ilog2ceil(n); // round up to power of two for scanning
    int* dev_bools;
    cudaMalloc((void**)&dev_bools, extended_n * sizeof(*dev_bools));
    checkCUDAError("cudaMalloc dev_buffer failed!");
    // fill zero and copy to boolean buffer
    cudaMemset(dev_bools, 0, extended_n * sizeof(*idata));

    auto block_size_booleanize = Common::getMapToBooleanBlockSize();
    auto full_blocks_per_grid_booleanize = fullBlocksPerGrid(n, block_size_booleanize);
    auto block_size_up = getUpSweepBlockSize();
    auto full_blocks_per_grid_up = fullBlocksPerGrid(extended_n, block_size_up);
    auto block_size_down = getDownSweepBlockSize();
    auto full_blocks_per_grid_down = fullBlocksPerGrid(extended_n, block_size_down);
    auto block_size_scatter = Common::getScatterBlocksize();
    auto full_blocks_per_grid_scatter = fullBlocksPerGrid(n, block_size_scatter);

    //// map to boolean
    //Common::kernMapToBoolean<<<full_blocks_per_grid_booleanize, block_size_booleanize >>>(n, );

    //// up sweep
    //auto pass_count = ilog2ceil(extended_n) - 1;
    //for (int d = 0; d <= pass_count; d++)
    //{
    //    kernScanUpSweepPass << <full_blocks_per_grid_up, block_size_up >> >(extended_n, 1 << d, dev_buffer);
    //}

    //// swap the last element of up sweep result and the real last element (0)
    //kernSwap << <1, 1 >> >(extended_n - 1, extended_n_plus_1 - 1, dev_buffer);

    //// down sweep
    //for (int d = pass_count; d >= 0; d--)
    //{
    //    kernScanDownSweepPass << <full_blocks_per_grid_down, block_size_down >> >(extended_n, 1 << d, dev_buffer);
    //}

    //// copy with offset to make it an inclusive scan
    //cudaMemcpy(odata, dev_buffer + 1, n * sizeof(*odata), copy_direction_to_odata);

    //cudaFree(dev_buffer);
}

}
}
