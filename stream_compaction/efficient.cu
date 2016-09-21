#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__
__global__ void upSweep(const int n, const int step, int *idata, int *odata) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= n) {
		return;
	}

	int rIndex = n - 1 - index;
	int mask = 1;

	for (int i = 1; i != step; (i <<= 1), (mask = mask << 1 | 1));

	if (index - step < 0 || (rIndex & mask) != 0) {
		odata[index] = idata[index];
	} else {
		odata[index] = idata[index] + idata[index - step];
	}
}

__global__ void downSweep(const int n, const int step, int *idata, int *odata) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= n) {
		return;
	}

	int rIndex = n - 1 - index;
	int mask = 1;

	for (int i = 1; i != step; (i <<= 1), (mask = mask << 1 | 1));

	if ((rIndex & mask) == 0 && index - step >= 0) {
		odata[index] = idata[index] + idata[index - step];
		odata[index - step] = idata[index];
	} else if (rIndex - step < 0 || (rIndex - step & mask) != 0) {
		odata[index] = idata[index];
	}
}

inline void swap(int &a, int &b) {
	auto tmp = a; a = b; b = tmp;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    // printf("TODO\n");
	int *dev_data[2];
	int input = 1;
	int output = 0;

	// device memory allocation
	cudaMalloc((void**)&dev_data[0], sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_data[0]");

	cudaMalloc((void**)&dev_data[1], sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_data[1]");

	// copy input data to device
	cudaMemcpy((void*)dev_data[input], (const void*)idata, sizeof(int) * n,
			cudaMemcpyHostToDevice);

	// do scan
	dim3 blockCount = (n - 1) / BLOCK_SIZE + 1;
	int step;

	swap(input, output);
	// up-sweep
	for (step = 1; step < n; step <<= 1) {
		swap(input, output);
		upSweep<<<blockCount, BLOCK_SIZE>>>(n, step, dev_data[input], dev_data[output]);
	}

	// set last element to 0
	cudaMemset(&dev_data[output][n - 1], 0, sizeof(int));

	// down-sweep
	for (step >>= 1; step > 0; step >>= 1) {
		swap(input, output);
		downSweep<<<blockCount, BLOCK_SIZE>>>(n, step, dev_data[input], dev_data[output]);
	}

	// copy result to host
	cudaMemcpy((void*)odata, (const void*)dev_data[output], sizeof(int) * n,
			cudaMemcpyDeviceToHost);

	// free memory on device
	cudaFree(dev_data[0]);
	cudaFree(dev_data[1]);
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
    // TODO
    return -1;
}

}
}
