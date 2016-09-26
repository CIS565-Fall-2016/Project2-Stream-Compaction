#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
	namespace Naive {

		// TODO: __global__ : finished

		__global__ void kernScan(int N, int start_idx, int *odata, const int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;
			if (index >= start_idx)
			{
				odata[index] = idata[index - start_idx] + idata[index];
			}
			else
			{
				odata[index] = idata[index];
			}
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			// TODO : finished
			int blockSize(128);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int *tmp_data, *tmp_data2;
			cudaMalloc((void**)&tmp_data, n * sizeof(int));
			cudaMalloc((void**)&tmp_data2, n * sizeof(int));
			cudaMemcpy(tmp_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int loop_times = ilog2ceil(n);
			int start_idx = 1;
			for (int i = 1; i <= loop_times; ++i)
			{
				kernScan<<<fullBlocksPerGrid, blockSize>>>(n, start_idx, tmp_data2, tmp_data);
				int *tmp_pt = tmp_data;
				tmp_data = tmp_data2;
				tmp_data2 = tmp_data;
				start_idx *= 2;
			}

			cudaMemcpy(odata, tmp_data, n * sizeof(int), cudaMemcpyDeviceToHost);
		}

	}
}
