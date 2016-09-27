#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
namespace Naive {

__global__ void kernNaiveScan(int n, int round, int * odata, int * idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	odata[index] = (
		(index < (1 << (round - 1))) 
			? 0 
			: idata[index - (1 << (round - 1))]
		) + idata[index];
}

__global__ void kernInclusiveToExclusiveScan(int n, int * odata, int * idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	odata[index] = (index == 0 ) ? 0 : idata[index - 1];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	int * dev_data;
	int * dev_data2;
	cudaMalloc((void**)&dev_data, n * sizeof(int));
	cudaMalloc((void**)&dev_data2, n * sizeof(int));
	cudaMemcpy((void*)dev_data, (void*)idata, n * sizeof(int), cudaMemcpyHostToDevice);

#if TIMING == 1
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif

	for (int i = 1; i <= ilog2ceil(n); i++) {
		kernNaiveScan << <fullBlocksPerGrid, blockSize >> >(n, i, dev_data2, dev_data);
		int * tempPtr = dev_data;
		dev_data = dev_data2;
		dev_data2 = tempPtr;
	}
	kernInclusiveToExclusiveScan << <fullBlocksPerGrid, blockSize >> >(n, dev_data2, dev_data);

#if TIMING == 1
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Naive scan: %f milliseconds\n", milliseconds);
#endif

	cudaMemcpy((void*)odata, (void*)dev_data2, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_data);
	cudaFree(dev_data2);
}

}
}
