#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {
	
const int BlockSize = StreamCompaction::Common::BlockSize;


static StreamCompaction::Common::Timer timer;

// naive scan on GPU
__global__ void scan(int n, int offset, int *input, int *output)
{
	int k = threadIdx.x + (blockDim.x * blockIdx.x);
	if (k >= n)
		return;
	
	if (k >= offset)
	{
		output[k] = input[k - offset] + input[k];
	}
	else
	{
		output[k] = input[k];
	}
}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	int *dev_in, *dev_out;

	cudaMalloc((void**)&dev_in, n * sizeof(int));
	checkCUDAError("cudaMalloc dev1 failed");

	cudaMalloc((void**)&dev_out, n * sizeof(int));
	checkCUDAError("cudaMalloc dev2 failed");

	cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	timer.startGpuTimer();

	dim3 fulllBlocksPerGrid((n + BlockSize - 1)/BlockSize);

	int maxD = ilog2ceil(n);
	int offset;
	for (int d = 1; d <= maxD; ++d)
	{
		//2^(d-1)
		offset = (1 << (d - 1));
		scan << < fulllBlocksPerGrid, BlockSize >> >(n, offset, dev_in, dev_out);

		int *tmp = dev_in;
		dev_in = dev_out;
		dev_out = tmp;
	}
	// last swap back in and out buffer
	int *tmp = dev_in;
	dev_in = dev_out;
	dev_out = tmp;

	timer.stopGpuTimer();
	timer.printTimerInfo("Scan::GPU::Naive = ", timer.getGpuElapsedTime());

	// exclusive scan
	cudaMemcpy(odata + 1, dev_out, (n-1)*sizeof(int), cudaMemcpyDeviceToHost);
	odata[0] = 0;
	
	cudaFree(dev_in);
	cudaFree(dev_out);

	checkCUDAError("naive scan error return");

}

}
}
