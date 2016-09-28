#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

__global__ void sum(int n, int startIndex, int *odata, const int *idata) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= n) return;
	if (index >= startIndex) {
		odata[index] = idata[index - startIndex] + idata[index];
	}
	else {
		odata[index] = idata[index];
	}
}

__global__ void inclusiveToExclusiveScan(int n, int *odata, const int *idata) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n) {
		odata[index] = index == 0 ? 0 : idata[index - 1];
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
float scan(int n, int *odata, const int *idata) {
	int blockSize = 128;
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	int* dev_idata;
	int* dev_odata;
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_idata failed!");
	
	cudaMalloc((void**)&dev_odata, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_odata failed!");

	cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	int numLevels = ilog2ceil(n);
	for (int startIndex = 1; startIndex <= (1 << (numLevels - 1)); startIndex *= 2) {
		sum << <fullBlocksPerGrid, blockSize >> >(n, startIndex, dev_odata, dev_idata);
		std::swap(dev_idata, dev_odata);
	}

	inclusiveToExclusiveScan << <fullBlocksPerGrid, blockSize >> >(n, dev_odata, dev_idata);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

	cudaFree(dev_idata);
	cudaFree(dev_odata);

	return milliseconds;
}

}
}
