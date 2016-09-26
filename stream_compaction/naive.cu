#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <utility>

#define blockSize 128

namespace StreamCompaction {
namespace Naive {

// TODO: __global__


__global__ void kernelScan(int offset, int n, int *swapA, int *swapB) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	if (index < offset) {
		swapB[index] = swapA[index];
	}
	else {
		swapB[index] = swapA[index - offset] + swapA[index];
	}
}
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	int offset;
	int *swapA, *swapB;
	cudaMalloc((void**)&swapA, n * sizeof(int));
	checkCUDAError("cudaMalloc swapA failed!");
	cudaMalloc((void**)&swapB, n * sizeof(int));
	checkCUDAError("cudaMalloc swapB failed!");

	cudaMemcpy(swapA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
	
	int log2n = ilog2ceil(n);
	for (int i = 1; i <= log2n; i++) {
		offset = 1 << (i - 1);
		kernelScan << <fullBlocksPerGrid, blockSize>> >(offset, n, swapA, swapB);
		std::swap(swapA, swapB);
	}
	cudaMemcpy(odata + 1, swapA, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
	odata[0] = 0;

	cudaFree(swapA);
	cudaFree(swapB);
}

}
}
