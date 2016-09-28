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
		float scan(int n, int *odata, const int *idata, int blockSize) {
			// TODO : finished
			// record time
			float diff(0);
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int *tmp_data, *tmp_data2;
			cudaMalloc((void**)&tmp_data, n * sizeof(int));
			cudaMalloc((void**)&tmp_data2, n * sizeof(int));
			cudaMemset(tmp_data2, 0, n * sizeof(int));
			cudaMemset(tmp_data, 0, n * sizeof(int));
			cudaMemcpy(tmp_data+1, idata, (n-1) * sizeof(int), cudaMemcpyHostToDevice);
			int loop_times = ilog2ceil(n);
			int start_idx = 1;
			for (int i = 0; i < loop_times; ++i)
			{
				kernScan<<<fullBlocksPerGrid, blockSize>>>(n, start_idx, tmp_data2, tmp_data);
				int *tmp_pt = tmp_data;
				tmp_data = tmp_data2;
				tmp_data2 = tmp_pt;
				start_idx *= 2;
			}

			cudaMemcpy(odata, tmp_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(tmp_data);
			cudaFree(tmp_data2);

			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&diff, start, end);

			return diff;
		}

	}
}
