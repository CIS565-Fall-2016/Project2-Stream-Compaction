#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128
namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void kernUpSweep(int n, int offset, int *odata, const int *idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	odata[index + offset - 1] += 
}

__global__ void kernDownSweep(int n, int offset, int *odata, const int *idata) {

}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	int *buf;
	cudaMalloc((void**)&buf, n * sizeof(int));
	checkCUDAError("cudaMalloc buf failed!");

	int offset;
	for (int i = 0; i <= ilog2(n); i++) {
		kernUpSweep << <fullBlocksPerGrid, blockSize >> >(n, offset, odata, idata);
	}
	for (int i = ilog2(n); i <= 0; i--) {
		kernDownSweep << <fullBlocksPerGrid, blockSize >> >(n, offset, odata, idata);
	}

	cudaFree(buf);
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
