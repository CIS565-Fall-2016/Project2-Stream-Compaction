#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "timer.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

	__global__ void kernSumUp(int N, int inStartIdx, int* inArray, int* outArray)
	{
		const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
		if (inStartIdx <= iSelf && iSelf < N)
		{
			outArray[iSelf] = inArray[iSelf - inStartIdx] + inArray[iSelf];
		}
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
	void scan(int n, int *odata, const int *idata)
	{
		if (n <= 0 || odata == NULL || idata == NULL)
			return;

		const int blockSize = 96;
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

		int* cudaIn = NULL;
		cudaMalloc((void**)&cudaIn, n * sizeof(int));
		checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
		cudaMemcpy(cudaIn, idata, n * sizeof(int), cudaMemcpyHostToDevice);

		int* cudaOut = NULL;
		cudaMalloc((void**)&cudaOut, n * sizeof(int));
		checkCUDAErrorWithLine("cudaMalloc cudaOut failed!");

		int maxDepth = ilog2ceil(n);
		Timer::playTimer();
		for (int d = 0; d < maxDepth; ++d)
		{
			const int inStartIdx = 1 << d;
			cudaMemcpy(cudaOut, cudaIn, inStartIdx * sizeof(int), cudaMemcpyHostToDevice);

			kernSumUp << <fullBlocksPerGrid, blockSize >> >(n, inStartIdx, cudaIn, cudaOut);

			// Ping-pong the buffers
			int* cudaTemp = cudaIn; cudaIn = cudaOut; cudaOut = cudaTemp;
		}
		Common::convertInclusiveToExclusiveScan << <fullBlocksPerGrid, blockSize >> > (n, cudaIn, cudaOut);
		Timer::pauseTimer();

		cudaMemcpy(odata, cudaOut, n * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(cudaOut);
		cudaFree(cudaIn);
	}

}
}
