#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

	__global__ void kernDownSweep(int offset, int n, int* odata, const int* idata) {
		int index = threadIdx.x + (blockDim.x * blockIdx.x);

		if (index >= n) return;
		int temp = odata[index + (1 << offset) - 1];
		odata[index + (1 << offset) - 1] = odata[index + (1 << (offset + 1)) - 1];
		odata[index + (1 << (offset + 1)) - 1] += temp;
	}

	__global__ void kernReduce(int d, int n, int* odata, const int* idata) {
		int index = threadIdx.x + (blockDim.x * blockIdx.x);

		if (index >= n) {
			return;
		}
		//int some = index + (1 << (d + 1)) - 1;
		//odata[index] = 32;
		odata[index] = index + (1 << (d + 1)) - 1;
		//some = some + 1;
		//odata[index + (1 << (d + 1)) - 1] = idata[index + (1 << (d)) - 1] + idata[index + (1 << (d + 1)) - 1];
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	
	int *dev_out;
	int *dev_in;

	cudaMalloc((void**)&dev_in, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_in failed!");
	cudaMalloc((void**)&dev_out, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_out failed!");

	cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	//up-sweep
	for (int d = 0; d < ilog2ceil(n); ++d) {
		kernReduce << <fullBlocksPerGrid, blockSize >> >(d, n, dev_out, dev_in);
	}
	
	//down-sweep
	/*odata[n - 1] = 0;
	for (int d = ilog2ceil(n) - 1; d > 0; --d) {
		kernDownSweep << <fullBlocksPerGrid, blockSize >> >(d, n, dev_out, dev_in);
	}*/
	cudaMemcpy(odata, dev_out, sizeof(int) * (n), cudaMemcpyDeviceToHost);
	for (int j = 0; j < n; ++j) {
		printf("%d\n", odata[j]);
	}
	cudaFree(dev_in);
	cudaFree(dev_out);
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
