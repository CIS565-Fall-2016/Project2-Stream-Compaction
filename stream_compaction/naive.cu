#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

	__global__ void parallelAdd(int n, int depth, int* odata, int* idata)
{
	int k = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (k >= n)
		return;

	if (k < (int)powf(2, depth - 1))
		odata[k] = idata[k];
	else
		odata[k] = idata[k - (int)powf(2, depth - 1)] + idata[k];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    
	dim3 numBlocks = (n + blocksize - 1) / blocksize;

	int * dev_beforeScan;
	int * dev_afterScan;

	cudaMalloc((void **)&dev_beforeScan, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_beforeScan failed!");
	cudaMalloc((void **)&dev_afterScan, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_afterScan failed!");

	cudaMemcpy(dev_beforeScan, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy idata to dev_afterScan failed!");

	//for d = 1 to lg(n) do
	for (int depth = 1; depth <= ilog2ceil(n); ++depth)
	{
		parallelAdd << <numBlocks, blocksize >> >(n, depth, dev_afterScan, dev_beforeScan);

		//ping-pong buffers
		int * temp = dev_afterScan;
		dev_afterScan = dev_beforeScan;
		dev_beforeScan = temp;
	}

	//because of ping-ponging, last iteration will be stored in beforeScan
	//additionally, we need to convert the inclusive scan into an exclusive scan
	odata[0] = 0;
	cudaMemcpy(&odata[1], dev_beforeScan, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_beforeScan to odata failed!");

	cudaFree(dev_beforeScan);
	cudaFree(dev_afterScan);
}

}
}


