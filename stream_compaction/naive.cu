#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#define MEASURE_EXEC_TIME
#include "naive.h"

namespace StreamCompaction {
	namespace Naive {

		__global__ void kernScanOneLevel(int stride, int n, int * __restrict__ odata, const int * __restrict__ idata)
		{
			unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

			if (tid >= n) return;

			if (tid >= stride)
			{
				odata[tid] = idata[tid - stride] + idata[tid];
			}
			else
			{
				odata[tid] = idata[tid];
			}
		}

		__global__ void makeExclusive(int n, int * __restrict__ odata, int * __restrict__ idata)
		{
			unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

			if (tid >= n) return;

			odata[tid] = tid == 0 ? 0 : idata[tid - 1];
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
#ifdef MEASURE_EXEC_TIME
		float scan(int n, int *odata, const int *idata)
		{
			if (n <= 0 || !odata || !idata || odata == idata)
			{
				return -1;
			}
#else
		void scan(int n, int *odata, const int *idata)
		{
			if (n <= 0 || !odata || !idata || odata == idata)
			{
				return;
			}
#endif
			const size_t kArraySizeInByte = n * sizeof(int);
			int *idata_dev = nullptr, *odata_dev = nullptr;

			cudaMalloc(&idata_dev, kArraySizeInByte);
			cudaMalloc(&odata_dev, kArraySizeInByte);
			cudaMemcpy(idata_dev, idata, kArraySizeInByte, cudaMemcpyHostToDevice);

#ifdef MEASURE_EXEC_TIME
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
#endif

			const int numLevels = ilog2ceil(n);
			const int threadsPerBlock = 256;
			const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
			int stride = 1;

			for (int i = 0; i < numLevels; ++i)
			{
				kernScanOneLevel << <numBlocks, threadsPerBlock >> >(stride, n, odata_dev, idata_dev);
				stride <<= 1;
				//swap
				int *tmp = odata_dev;
				odata_dev = idata_dev;
				idata_dev = tmp;
			}
			makeExclusive << <numBlocks, threadsPerBlock >> >(n, odata_dev, idata_dev);

#ifdef MEASURE_EXEC_TIME
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float millisceconds = 0;
			cudaEventElapsedTime(&millisceconds, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
#endif

			cudaMemcpy(odata, odata_dev, kArraySizeInByte, cudaMemcpyDeviceToHost);
			cudaFree(idata_dev);
			cudaFree(odata_dev);
			cudaDeviceSynchronize(); // make sure result is ready

#ifdef MEASURE_EXEC_TIME
			return millisceconds;
#endif
		}

	}
}
