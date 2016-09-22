#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
	namespace radixSort {

		__global__ void makeBarray(int n, int *odata, int *idata, int d) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;
			int val = idata[index] & d;
			odata[index] = val ? 1 : 0;
		}

		__global__ void makeEarray(int n, int *odata, int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;
			odata[index] = idata[index] ? 0 : 1;
		}

		__global__ void makeTarray(int n, int *odata, int *idata,int totalFalse) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;
			odata[index] = index - idata[index] + totalFalse;
		}

		__global__ void makeDarray(int n, int *odata, int *b, int *t, int *f) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;
			odata[index] = b[index] ? t[index] : f[index];
		}

		__global__ void reorder(int n, int *odata, int *idata, int *d) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;
			odata[d[index]] = idata[index];
		}

		void sort(int n, int *odata, const int *idata) {
			int *dev_idata;
			int *dev_odata;
			int *dev_b;
			int *dev_f;
			int *dev_t;
			int *dev_d;

			int last_e;
			int last_f;

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("Cannot allocate memory for idata");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("Cannot allocate memory for odata");
			cudaMalloc((void**)&dev_b, n * sizeof(int));
			checkCUDAError("Cannot allocate memory for odata");
			cudaMalloc((void**)&dev_f, n * sizeof(int));
			checkCUDAError("Cannot allocate memory for odata");
			cudaMalloc((void**)&dev_t, n * sizeof(int));
			checkCUDAError("Cannot allocate memory for odata");
			cudaMalloc((void**)&dev_d, n * sizeof(int));
			checkCUDAError("Cannot allocate memory for odata");

			cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			
			for (int i = 0; i < ilog2ceil(n); i++){
				makeBarray << <fullBlocksPerGrid,blockSize >> >(n, dev_b, dev_idata, 1 << i);
				makeEarray << <fullBlocksPerGrid, blockSize >> >(n, dev_f, dev_b);
				StreamCompaction::Efficient::scan(n, dev_f, dev_f);

				cudaMemcpy(&last_e, dev_b + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&last_f, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

				int totalFalses = (last_e ? 0 : 1) + last_f;
			
				makeTarray << <fullBlocksPerGrid, blockSize >> >(n, dev_t, dev_f, totalFalses);
				makeDarray << <fullBlocksPerGrid, blockSize >> >(n, dev_d, dev_b, dev_t, dev_f);
			
				reorder << <fullBlocksPerGrid, blockSize >> >(n, dev_odata, dev_idata, dev_d);

				int *temp = dev_odata;
				dev_odata = dev_idata;
				dev_idata = temp;
			}
			cudaMemcpy(odata, dev_idata, n*sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_b);
			cudaFree(dev_f);
			cudaFree(dev_t);
			cudaFree(dev_d);
		}

	}
}
