#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include "radix_sort.h"

namespace StreamCompaction {
	namespace RadixSort {

		__global__ void kernInitEBitMap(int N, int bit, int* inBuffer, int* outEBitMap)
		{
			const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
			if (0 <= iSelf && iSelf < N)
			{
				outEBitMap[iSelf] = 1 - ((inBuffer[iSelf] & (1 << bit)) != 0);
			}
		}

		__global__ void kernInitTArray(int N, int numTotalFalses, int* cudaFBuffer, int* outTBuffer)
		{
			const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
			if (0 <= iSelf && iSelf < N)
			{
				outTBuffer[iSelf] = iSelf - cudaFBuffer[iSelf] + numTotalFalses;
			}
		}

		__global__ void kernInitDArray(int N, int* cudaEBuffer, int* cudaTBuffer, int* cudaFBuffer, int* outDBuffer)
		{
			const int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
			if (0 <= iSelf && iSelf < N)
			{
				outDBuffer[iSelf] = cudaEBuffer[iSelf] ? cudaFBuffer[iSelf] : cudaTBuffer[iSelf];
			}
		}

		/**
		* Performs Parallel Radix Sort.
		*/
		void sort(int n, int *odata, const int *idata)
		{
			if (n <= 0 || odata == NULL || idata == NULL)
				return;

			const int blockSize = 128;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int* cudaInBuffer = NULL;
			cudaMalloc((void**)&cudaInBuffer, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc cudaInBuffer failed!");
			cudaMemcpy(cudaInBuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int* cudaEBuffer = NULL;
			cudaMalloc((void**)&cudaEBuffer, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc cudaEBuffer failed!");
			cudaMemset(cudaEBuffer, 0, n * sizeof(int));

			int* cudaFBuffer = NULL;

			int* cudaTBuffer = NULL;
			cudaMalloc((void**)&cudaTBuffer, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc cudaTBuffer failed!");
			cudaMemset(cudaTBuffer, 0, n * sizeof(int));

			int* cudaDBuffer = NULL;
			cudaMalloc((void**)&cudaDBuffer, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc cudaDBuffer failed!");
			cudaMemset(cudaDBuffer, 0, n * sizeof(int));

			int* cudaOutBuffer = NULL;
			cudaMalloc((void**)&cudaOutBuffer, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc cudaOutBuffer failed!");
			cudaMemset(cudaOutBuffer, 0, n * sizeof(int));

			const int maxNumBits = ilog2ceil(n) + 1;
			for (int bitIdx = 0; bitIdx < maxNumBits; ++bitIdx)
			{
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
				kernInitEBitMap << <fullBlocksPerGrid, blockSize >> > (n, bitIdx, cudaInBuffer, cudaEBuffer);
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
				cudaFBuffer = StreamCompaction::Efficient::scanInHostPlace(n, cudaEBuffer);
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");

				int numTotalFalses = 0;
				{
					checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
					int lastEElement = 0;
					checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
					cudaMemcpy(&lastEElement, cudaEBuffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
					checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
					int lastFElement = 0;
					checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
					cudaMemcpy(&lastFElement, cudaFBuffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
					numTotalFalses = lastEElement + lastFElement;
				}
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
				kernInitTArray << <fullBlocksPerGrid, blockSize >> > (n, numTotalFalses, cudaFBuffer, cudaTBuffer);
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
				kernInitDArray << <fullBlocksPerGrid, blockSize >> > (n, cudaEBuffer, cudaTBuffer, cudaFBuffer, cudaDBuffer);
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
				StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, cudaOutBuffer, cudaInBuffer, cudaDBuffer);
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");
				cudaMemcpy(odata, cudaOutBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(cudaInBuffer, cudaOutBuffer, n * sizeof(int), cudaMemcpyDeviceToDevice);
				checkCUDAErrorWithLine("cudaMalloc cudaIn failed!");

				cudaFree(cudaFBuffer);
			}
			

			cudaFree(cudaOutBuffer);
			cudaFree(cudaDBuffer);
			cudaFree(cudaTBuffer);
			cudaFree(cudaEBuffer);
			cudaFree(cudaInBuffer);

		}

	}
}
