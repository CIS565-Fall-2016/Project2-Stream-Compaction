#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <utility>

#define blockSize 128

namespace StreamCompaction {
namespace Naive {

// TODO: __global__


__global__ void kernelScan(int offset, int n, int *dev_odata, int *dev_idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	if (index < offset) {
		dev_idata[index] = dev_odata[index];
	}
	else {
		dev_idata[index] = dev_odata[index - offset] + dev_odata[index];
	}
}
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	int offset;
	int *dev_odata, *dev_idata;
	cudaMalloc((void**)&dev_odata, n * sizeof(int));
	cudaMalloc((void**)&dev_idata, n * sizeof(int));

	cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	int log2n = ilog2ceil(n);
	for (int i = 1; i <= log2n; i++) {
		offset = 1 << (i - 1);
		kernelScan << <fullBlocksPerGrid, blockSize>> >(offset, n, dev_odata, dev_idata);
		std::swap(dev_odata, dev_idata);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, end);
	printf("Naive scan: %f ms\n", milliseconds);

	cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
	odata[0] = 0;

	cudaFree(dev_odata);
	cudaFree(dev_idata);
}

}
}
