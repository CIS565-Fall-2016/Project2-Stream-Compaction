#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
	namespace Naive {

		// Done: __global__

		__global__ void kernReduce(int offset, int n, int *in, int *out) {
			int index = threadIdx.x + (blockDim.x * blockIdx.x);

			if (index >= n) 
				return;

			if (index >= offset) {
				out[index] = in[index] + in[index - offset];
			}
			else {
				out[index] = in[index];
			}

		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata, float& time) {
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int *dev_out;
			int *dev_in;

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_in failed!");
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_out failed!");

			cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			
			float milliseconds = 0;
			cudaEventRecord(start);
			for (int d = 1; d <= ilog2ceil(n); ++d) {
				kernReduce << <fullBlocksPerGrid, blockSize >> >((1 << (d - 1)), n, dev_in, dev_out);
				std::swap(dev_in, dev_out);
			}
			cudaEventRecord(stop);
			std::swap(dev_in, dev_out);
			cudaMemcpy(odata + 1, dev_out, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
			
			odata[0] = 0;
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			time = milliseconds;
			//printf("%f\n", milliseconds);
			cudaFree(dev_out);
			cudaFree(dev_in);
		}

	}
}
