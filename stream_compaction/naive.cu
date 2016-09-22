#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
namespace Naive {

	__global__ void kernRunScan(int N, int pow2d, int* odata, int* idata) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index > N || index < 0) {
			return;
		}
		
		if (index >= pow2d) {
			odata[index] = idata[index - pow2d] + idata[index];
		}
		else {
			odata[index] = idata[index];
		}
	}

	__global__ void kernInclusiveToExclusive(int N, int* odata, int* idata) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index > N) {
			return;
		}

		if (index == 0) {
			odata[0] = 0;
		}
		else {
			odata[index] = idata[index - 1];
		}
	}
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	dim3 threadsPerBlock(blockSize);

	// Move data GPU-side
	int* dev_in;
	int* dev_out;

	cudaMalloc((void**)&dev_in, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_in.");

	cudaMalloc((void**)&dev_out, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_out.");

	cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	
	int max_d = ilog2ceil(n);

	// Loop over data 
	for (int d = 1; d <= max_d; d++) {

		kernRunScan << < fullBlocksPerGrid, threadsPerBlock >> >(n, pow(2, d - 1), dev_out, dev_in);

		int* swap = dev_out;
		dev_out = dev_in;
		dev_in = swap;
	}

	kernInclusiveToExclusive << < fullBlocksPerGrid, threadsPerBlock >> >(n, dev_out, dev_in);

	cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy back failed!");

	cudaFree(dev_in);
	cudaFree(dev_out);
}

}
}
