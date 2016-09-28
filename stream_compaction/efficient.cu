#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <chrono>

namespace StreamCompaction {
	namespace Efficient {

		// TODO: __global__
		__global__ void kernScanUpSweep(int N, int interval, int *data)
		{
			// up sweep
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int real_index = index * interval * 2;
			if (real_index >= N) return;
			int cur_index = real_index + 2 * interval - 1;
			int last_index = real_index + interval - 1;
			if (cur_index >= N) return;

			data[cur_index] = data[last_index] + data[cur_index];
		}

		__global__ void kernScanDownSweep(int N, int interval, int *data)
		{
			// down seep
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int real_index = index * interval * 2;
			if (real_index >= N) return;
			int last_index = real_index + interval - 1;
			int cur_index = real_index + 2 * interval - 1;
			if (cur_index >= N) return;
			int tmp = data[last_index];
			data[last_index] = data[cur_index];
			data[cur_index] += tmp;
		}

		__global__ void kernMapDigitToBoolean(int N, int digit, int *odata, const int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;
			int mask = 1 << digit;
			if ((idata[index] & mask) == 0)
			{
				if (digit != 31) odata[index] = 0;
				else odata[index] = 1;
			}
			else
			{
				if (digit != 31) odata[index] = 1;
				else odata[index] = 0;
			}
		}

		__global__ void kernFlipBoolean(int N, int *odata, const int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;
			if (idata[index] == 0)
			{
				odata[index] = 1;
			}
			else
			{
				odata[index] = 0;
			}
		}

		__global__ void kernSortOneRound(int N, int *bools, int *indices_zero, int *indices_one, int maxFalse,
			int *odata, const int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;
			if (bools[index] == 0)
			{
				// false;
				odata[indices_zero[index]] = idata[index];
			}
			else
			{
				odata[indices_one[index] + maxFalse] = idata[index];
			}
		}

		void radix_sort(int n, int *odata, const int *idata, int blockSize)
		{
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int *cuda_idata, *cuda_bools_one, *cuda_bools_zero, 
				*cuda_indices_one, *cuda_indices_zero, *cuda_odata;
			int *bools = new int[n];
			int *indices = new int[n];

			cudaMalloc((void**)&cuda_idata, n * sizeof(int));
			cudaMalloc((void**)&cuda_bools_one, n * sizeof(int));
			cudaMalloc((void**)&cuda_bools_zero, n * sizeof(int));
			cudaMalloc((void**)&cuda_odata, n * sizeof(int));
			cudaMalloc((void**)&cuda_indices_one, n * sizeof(int));
			cudaMalloc((void**)&cuda_indices_zero, n * sizeof(int));

			cudaMemcpy(cuda_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);

			for (int i = 0; i < 32; ++i)
			{
				kernMapDigitToBoolean << <fullBlocksPerGrid, blockSize >> >(n, i, cuda_bools_one, cuda_idata);
				cudaMemcpy(bools, cuda_bools_one, n * sizeof(int), cudaMemcpyDeviceToHost);
				scan(n, indices, bools);
				cudaMemcpy(cuda_indices_one, indices, n * sizeof(int), cudaMemcpyHostToDevice);

				kernFlipBoolean << <fullBlocksPerGrid, blockSize >> >(n, cuda_bools_zero, cuda_bools_one);
				cudaMemcpy(bools, cuda_bools_zero, n * sizeof(int), cudaMemcpyDeviceToHost);
				scan(n, indices, bools);
				cudaMemcpy(cuda_indices_zero, indices, n * sizeof(int), cudaMemcpyHostToDevice);

				int totalFalse;
				cudaMemcpy(&totalFalse, &cuda_indices_zero[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
				totalFalse += bools[n - 1];

				kernSortOneRound << <fullBlocksPerGrid, blockSize >> >(n, cuda_bools_one, cuda_indices_zero, cuda_indices_one, totalFalse, cuda_odata, cuda_idata);

				int *tmp = cuda_idata;
				cuda_idata = cuda_odata;
				cuda_odata = tmp;
			}

			cudaMemcpy(odata, cuda_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(cuda_idata);
			cudaFree(cuda_bools_one);
			cudaFree(cuda_bools_zero);
			cudaFree(cuda_odata);
			cudaFree(cuda_indices_one);
			cudaFree(cuda_indices_zero);
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		float scan(int n, int *odata, const int *idata, int blockSize) {
			// record time
			float diff(0);
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);

			int loop_times = ilog2ceil(n);
			int totalNum = 1;
			for (int i = 0; i < loop_times; ++i)
			{
				totalNum *= 2;
			}
			int interval = 1;
			//printf("total looptimes: %d, total num %d\n", loop_times, totalNum);

			int *tmp_data;
			cudaMalloc((void**)&tmp_data, totalNum * sizeof(int));
			cudaMemset(tmp_data, 0, totalNum);
			cudaMemcpy(tmp_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			
			// up sweep
			for (int i = 0; i < loop_times; ++i)
			{
				dim3 fullBlocksPerGrid((totalNum / (interval * 2) + blockSize - 1) / blockSize);
				kernScanUpSweep << <fullBlocksPerGrid, blockSize >> >(totalNum, interval, tmp_data);
				interval *= 2;
			}

			// down sweep
			cudaMemset(&tmp_data[totalNum - 1], 0, sizeof(int));

			for (int i = 0; i < loop_times; ++i)
			{
				dim3 fullBlocksPerGrid((totalNum / interval + blockSize - 1) / blockSize);
				interval /= 2;
				kernScanDownSweep << <fullBlocksPerGrid, blockSize >> >(totalNum, interval, tmp_data);
			}

			cudaMemcpy(odata, tmp_data, n*sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(tmp_data);

			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&diff, start, end);

			//printf("GPU scan took %fms\n", diff);
			return diff;
		}

		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param odata  The array into which to store elements.
		 * @param idata  The array of elements to compact.
		 * @returns      The number of elements remaining after compaction.
		 */
		int compact(int n, int *odata, const int *idata, double &time, int blockSize) {
			// TODO
			// record time
			float diff(0);
			cudaEvent_t start, end;
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int *indices_cuda;
			int *bools_cuda;
			int *idata_cuda;
			int *odata_cuda;

			int *indices = new int[n];
			int *bools = new int[n];

			cudaMalloc((void**)&indices_cuda, n * sizeof(int));
			cudaMalloc((void**)&bools_cuda, n * sizeof(int));
			cudaMalloc((void**)&idata_cuda, n * sizeof(int));
			cudaMalloc((void**)&odata_cuda, n * sizeof(int));

			cudaMemcpy(idata_cuda, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, bools_cuda, idata_cuda);

			cudaMemcpy(bools, bools_cuda, n * sizeof(int), cudaMemcpyDeviceToHost);

			scan(n, indices, bools);

			cudaMemcpy(indices_cuda, indices, n * sizeof(int), cudaMemcpyHostToDevice);

			Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, odata_cuda, idata_cuda, bools_cuda, indices_cuda);

			int remain_elem;
			cudaMemcpy(&remain_elem, &indices_cuda[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
			remain_elem += bools[n - 1];
			//for (int i = 0; i < n; ++i)
			//{
			//	if (bools[i] == 1) remain_elem++;
			//}

			cudaMemcpy(odata, odata_cuda, remain_elem * sizeof(int), cudaMemcpyDeviceToHost);

			delete[] bools;
			delete[] indices;

			cudaFree(indices_cuda);
			cudaFree(bools_cuda);
			cudaFree(idata_cuda);
			cudaFree(odata_cuda);

			cudaEventRecord(end, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&diff, start, end);

			//printf("GPU compact took %fms\n", diff);

			time = diff;
			return remain_elem;
		}

	}
}
