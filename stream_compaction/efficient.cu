#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "timer.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

	__global__ void kernSumUpSweep(int N, size_t clusterSize, int* buffer)
	{
		const unsigned long iSelf = clusterSize *  (threadIdx.x + (blockIdx.x * blockDim.x));
		const unsigned long clusterBitMask = clusterSize - 1;
		if (0 <= iSelf && iSelf < N && ( (iSelf & clusterBitMask) == 0) )
		{
			buffer[iSelf + clusterSize - 1] += buffer[iSelf + (clusterSize >> 1) - 1];
		}
	}

	__global__ void kernSumDownSweep(int N, size_t clusterSize, int* buffer)
	{
		const unsigned long iSelf = clusterSize * ( threadIdx.x + (blockIdx.x * blockDim.x) );
		const unsigned long clusterBitMask = clusterSize - 1;
		if (0 <= iSelf && iSelf < N && ((iSelf & clusterBitMask) == 0))
		{
			int leftChildVal = buffer[iSelf + (clusterSize >> 1) - 1]; 
			buffer[iSelf + (clusterSize >> 1) - 1] = buffer[iSelf + clusterSize - 1]; // Set left child to this node’s value
			buffer[iSelf + clusterSize - 1] += leftChildVal; // Set right child to old left value + this node’s value
		}
	}


	void _scanInHostPlace(int n, int *cudaBuffer, const int *idata)
	{
		const int blockSize =  128;

		const int maxDepth = ilog2ceil(n);
		const int nextPowerOf2 = 1 << maxDepth;

		cudaMemset(cudaBuffer, 0, nextPowerOf2 * sizeof(int));
		cudaMemcpy(cudaBuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

		n = nextPowerOf2;

		Timer::playTimer();
		int d = 0;
		for (; d < maxDepth; ++d)
		{
			const size_t clusterSize = 1 << (d + 1);
			const size_t numThreads = (n >= clusterSize) ? n / clusterSize : n;

			dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
			kernSumUpSweep << <fullBlocksPerGrid, blockSize >> >(n, clusterSize, cudaBuffer);
		}

		cudaMemset(cudaBuffer + (n - 1), 0, sizeof(int));
		for (; d >= 0; --d)
		{
			const size_t clusterSize = 1 << (d + 1);
			const size_t numThreads = (n >= clusterSize) ? n / clusterSize : n;

			dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
			kernSumDownSweep << <fullBlocksPerGrid, blockSize >> >(n, clusterSize, cudaBuffer);
		}
		Timer::pauseTimer();
	}

int* scanInHostPlace(int n, const int *idata)
{
	if (n <= 0 || idata == NULL)
		return NULL;
	checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
	const int maxDepth = ilog2ceil(n);
	const int nextPowerOf2 = 1 << maxDepth;

	int* cudaBuffer = NULL;
	checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
	cudaMalloc((void**)&cudaBuffer, nextPowerOf2 * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
	_scanInHostPlace(n, cudaBuffer, idata);

	return cudaBuffer;
}

/**
* Performs prefix-sum (aka scan) on idata, storing the result into odata.
*/

void scan(int n, int *odata, const int *idata)
{
	if (n <= 0 || odata == NULL || idata == NULL)
		return;

	int* cudaBuffer = scanInHostPlace(n, idata);

	cudaMemcpy(odata, cudaBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cudaBuffer);
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
int compact(int n, int *odata, const int *idata)
{
	if (n <= 0 || odata == NULL || idata == NULL)
		return 0;
	
	const int blockSize = 128;
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	int* cudaInBuffer = NULL;
	cudaMalloc((void**)&cudaInBuffer, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
	cudaMemcpy(cudaInBuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

	int* cudaBitMask = NULL;
	cudaMalloc((void**)&cudaBitMask, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
	cudaMemset(cudaBitMask, 0, n * sizeof(int));

	Timer::playTimer();

	Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, cudaBitMask, cudaInBuffer);
	cudaMemcpy(odata, cudaBitMask, n * sizeof(int), cudaMemcpyDeviceToHost);
	int endsWith1;
	cudaMemcpy(&endsWith1, &cudaBitMask[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

	int* cudaScanResult = scanInHostPlace(n, cudaBitMask);
	cudaMemcpy(odata, cudaScanResult, n * sizeof(int), cudaMemcpyDeviceToHost);
	
	int outNumElements = 0;
	cudaMemcpy(&outNumElements, cudaScanResult + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	outNumElements += endsWith1;

	int* cudaOutBuffer = NULL;
	cudaMalloc((void**)&cudaOutBuffer, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc cudaOutBuffer failed!");
	cudaMemset(cudaOutBuffer, 0, n * sizeof(int));

	Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, cudaOutBuffer, cudaInBuffer, cudaBitMask, cudaScanResult);

	Timer::pauseTimer();

	cudaMemcpy(odata, cudaOutBuffer, outNumElements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cudaOutBuffer);
	cudaFree(cudaScanResult);
	cudaFree(cudaBitMask);
	cudaFree(cudaInBuffer);
	
	return outNumElements;
}

}
}
