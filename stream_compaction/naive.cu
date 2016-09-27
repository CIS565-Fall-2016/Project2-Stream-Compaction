#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>
#include <iostream>

#define blockSize 128

namespace StreamCompaction {
	namespace Naive {

		__global__ void kernScanInnerLoop(int n, int *odata, int *idata, int d){
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			if (index >= d)
				odata[index] = idata[index - d] + idata[index];
			else
				odata[index] = idata[index];

		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {

			//cudaEvent_t start, stop;
			//cudaEventCreate(&start);
			//cudaEventCreate(&stop);

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			// Create GPU array pointers
			int *dev_oData;
			int *dev_iData;

			// Allocate GPU space
			cudaMalloc((void**)&dev_oData, n * sizeof(int));
			cudaMalloc((void**)&dev_iData, n * sizeof(int));

			// Copy data to GPU
			cudaMemcpy(dev_iData, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Failed to copy dev_iData");
			cudaMemcpy(dev_oData, odata, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Failed to copy dev_oData");

			//cudaEventRecord(start);
			// Perform scan
			for (int x = 1; x < n; x *= 2) {
				kernScanInnerLoop << <fullBlocksPerGrid, blockSize >> >(n, dev_oData, dev_iData, x);
				std::swap(dev_oData, dev_iData);
			}
			//cudaEventRecord(stop);

			//cudaEventSynchronize(stop);
			//float milliseconds = 0;
			//cudaEventElapsedTime(&milliseconds, start, stop);
			//std::cout << milliseconds << std::endl;

			// Swap back
			std::swap(dev_oData, dev_iData);

			// Copy data back to CPU
			cudaMemcpy(odata, dev_oData, sizeof(int)*n, cudaMemcpyDeviceToHost);

			// Shift right
			for (int x = n - 1; x > 0; x--) odata[x] = odata[x - 1];
			odata[0] = 0;

			// Free memory on GPU and CPU
			cudaFree(dev_iData);
			cudaFree(dev_oData);
		}

	}
}
