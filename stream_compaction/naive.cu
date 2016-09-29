#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


namespace StreamCompaction {
	namespace Naive {

		__device__ int threadIndex() {
			return (blockIdx.x * blockDim.x) + threadIdx.x;
		}


		__global__ void kernAdd(int d, int n, int *odata, int *idata) {
			int index = threadIndex();
			if (index >= n) return;
			odata[index] = (index < d ? 0 : idata[index - d]) + idata[index];
		}

		__global__ void kernShiftRight(int n, int *odata, int *idata) {
			int index = threadIndex();
			if (index == 0) odata[0] = 0;
			if (index >= n) return;
			odata[index] = idata[index - 1];
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata) {
			// TODO
			int* dev_idata;
			int* dev_odata;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			////////////
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
			////////////////

			int numBlocks = getNumBlocks(blockSize, n);
			for (int d = 1; d < n * 2; d *= 2) {
				kernAdd << <numBlocks, blockSize >> >(d, n, dev_odata, dev_idata);

				int *swap = dev_idata;
				dev_idata = dev_odata;
				dev_odata = swap;
			}
			kernShiftRight << <numBlocks, blockSize >> >(n, dev_odata, dev_idata);

			///////////
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("%f\n", milliseconds);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			/////////

			cudaMemcpy(odata, dev_odata, n* sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;

			cudaFree(dev_idata);
			cudaFree(dev_odata);
		}

	}
}
