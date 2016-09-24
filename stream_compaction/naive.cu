#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

int* devIdata;
int* devOdata;

namespace StreamCompaction {
	namespace Naive {

		// TODO: __global__
		__global__ void kernelNaive(int n, int delta, const int *idata, int *odata) {
			int index = (blockIdx.x *blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			if (index - delta < 0) {
				odata[index] = idata[index];
			} else {
               	odata[index] = idata[index - delta] + idata[index];
			}
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			// TODO
			// printf("TODO\n");
			cudaMalloc((void**)&devIdata, n * sizeof(int));
			checkCUDAError("cudaMalloc devIdata failed");

			cudaMalloc((void**)&devOdata, n * sizeof(int));
			checkCUDAError("cudaMalloc devOdata failed");

			cudaMemcpy(devIdata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			
			//performance check... remember...

			int blockNum = (n + blockSize - 1) / blockSize;
			
			//Naive Parallel Scan
			int level = ilog2ceil(n);
			int delta;
			for (int d = 1; d <= level; d++) {
				// pow (2,d-1)
				// refer to slides 
				delta = (1 << (d - 1));
				kernelNaive << < blockNum, blockSize >> >(n, delta, devIdata, devOdata);
				std::swap(devIdata, devOdata);
			}
			// Think twice.............
			std::swap(devIdata, devOdata);
			
			// exclusive scan, set odata[0] = 0 seperately 
			cudaMemcpy(odata + 1, devOdata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;

			cudaFree(devIdata);
			cudaFree(devOdata);

			checkCUDAError("naice scan error...");
		}

	}
}
