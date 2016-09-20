#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

__global__ void kernScanStep(int n, int d, int *odata, const int *idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	
	int s = pow((double)2, (double)(d - 1));

	if (index >= s) {
		odata[index] = idata[index] + idata[index - s];
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

    // create device arrays
	int *dev_in;
	int *dev_out;

	cudaMalloc((void**)&dev_in, n*sizeof(int));
	cudaMalloc((void**)&dev_out, n*sizeof(int));

	cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	for (int d = 1; d <= ilog2ceil(n); d++) {
		kernScanStep << < fullBlocksPerGrid, blockSize >> >(n, d, dev_out, dev_in);
		cudaMemcpy(dev_in, dev_out, n*sizeof(int), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(&odata[1], dev_out, (n-1)*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_in);
	cudaFree(dev_out);
}

}
}

