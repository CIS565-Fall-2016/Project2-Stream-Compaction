#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "sort.h"
#include <algorithm>
#include <thrust/scan.h>
#include "thrust.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
	namespace Sort {

		// Done: __global__

		__global__ void kernComputeTArray(int n, int total_falses, int* f, int *t) {
			int index = threadIdx.x + (blockDim.x * blockIdx.x);

			if (index >= n) return;
			t[index] = index - f[index] + total_falses;
		}

		__global__ void kernComputeEArray(int n, int shift, int *e, int *in) {
			int index = threadIdx.x + (blockDim.x * blockIdx.x);

			if (index >= n) return;
			e[index] = (in[index] >> shift) & 1 ? 0 : 1;
		}

		__global__ void kernScatter(int n, int* e, int *t, int *f, int* dev_out, int* dev_in) {
			int index = threadIdx.x + (blockDim.x * blockIdx.x);

			if (index >= n) return;

			dev_out[!e[index] ? t[index] : f[index]] = dev_in[index];
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void sort(int n, int *odata, const int *idata, float& time) {
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int *dev_in;
			int *dev_out;
			int *e;
			int *f;
			int *t;

			//int *t_host = new int[n];

			int *e_host = new int[n];

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_in failed!");
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_out failed!");
			cudaMalloc((void**)&e, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc e failed!");
			cudaMalloc((void**)&f, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc f failed!");
			cudaMalloc((void**)&t, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc t failed!");

			cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			
			float milliseconds = 0, totalTime = 0.f;
			
			//max number allowed is ilog2ceil(n) - 1; for ex: if n == 8, max value any element can have is 7
			for (int lsb = 0; lsb < 3; ++lsb) {
				cudaEventRecord(start);
				//compute e array
				kernComputeEArray << <fullBlocksPerGrid, blockSize >> >(n, lsb, e, dev_in);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);
				totalTime += milliseconds;
				//scan e
				cudaMemcpy(e_host, e, sizeof(int) * (n), cudaMemcpyDeviceToHost);
				int total_falses = e_host[n - 1];
				thrust::exclusive_scan(e_host, e_host + n, e_host);
				total_falses += e_host[n - 1];
				cudaMemcpy(f, e_host, sizeof(int) * n, cudaMemcpyHostToDevice);
				cudaEventRecord(start);
				//compute t array
				kernComputeTArray << <fullBlocksPerGrid, blockSize >> >(n, total_falses, f, t);

				//scatter
				kernScatter << <fullBlocksPerGrid, blockSize >> >(n, e, t, f, dev_out, dev_in);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);
				totalTime += milliseconds;
				std::swap(dev_in, dev_out);
			}
			std::swap(dev_in, dev_out);
			cudaMemcpy(odata, dev_out, sizeof(int) * (n), cudaMemcpyDeviceToHost);
			
			time = totalTime;

			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(e);
			cudaFree(t);
			cudaFree(f);
			
			delete[] e_host;
		}

	}
}
