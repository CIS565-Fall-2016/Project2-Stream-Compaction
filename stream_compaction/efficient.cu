#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

	__global__ void kernDownSweep(int d, int n, int* idata) {
		int index = threadIdx.x + (blockDim.x * blockIdx.x);

		if (index >= n) return;
		
		int off_n = 1 << (d + 1);
		int off = (1 << d);
		if ((index % (off_n) == 0)) {
			int temp = idata[index + (1 << d) - 1];
			idata[index + off - 1] = idata[index + off_n - 1];
			idata[index + off_n - 1] += temp;	
		}
	}

	__global__ void kernUpSweep(int d, int n, int* idata) {
		int index = threadIdx.x + (blockDim.x * blockIdx.x);

		if (index >= n) return;
		int off_n = 1 << (d + 1);
		int off = (1 << d);
		if (index % off_n == 0)
			idata[index + off_n - 1] += idata[index + off - 1];
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata, float& time) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//for non power of 2
	int ilog = ilog2ceil(n);
	int off_n = 1 << ilog;

	dim3 fullBlocksPerGrid((off_n + blockSize - 1) / blockSize);
	
	int *dev_in;

	cudaMalloc((void**)&dev_in, off_n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_in failed!");

	cudaMemcpy(dev_in, idata, sizeof(int) * off_n, cudaMemcpyHostToDevice);
	float milliseconds = 0;
	cudaEventRecord(start);
	//up-sweep
	for (int d = 0; d < ilog; ++d) {
		kernUpSweep << <fullBlocksPerGrid, blockSize >> >(d, off_n, dev_in);
	}
	
	//set the last value as zero
	cudaMemset(dev_in + (off_n - 1), 0, sizeof(int));
	
	//down-sweep
	for (int d = ilog - 1; d >= 0; --d) {
		kernDownSweep << <fullBlocksPerGrid, blockSize >> >(d, off_n, dev_in);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	time = milliseconds;
	cudaMemcpy(odata, dev_in, sizeof(int) * (n), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_in);
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
int compact(int n, int *odata, const int *idata, float& time) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	int *bools;
	int *dev_in;
	int *dev_out;
	int *indices;
	int *tmp = new int[n];
	int *tmp_bools = new int[n];
	cudaMalloc((void**)&dev_in, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_in failed!");
	cudaMalloc((void**)&dev_out, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_out failed!");
	cudaMalloc((void**)&indices, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc indices failed!");
	cudaMalloc((void**)&bools, n * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc bools failed!");

	cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	float milliseconds = 0, totalTime = 0.f;
	cudaEventRecord(start);
	//map to boolean
	Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, bools, dev_in);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totalTime += milliseconds;
	int j = 0;
	//scan
	cudaMemcpy(odata, bools, sizeof(int) * (n), cudaMemcpyDeviceToHost);
	float time_scan;
	scan(n, tmp, odata, time_scan);
	totalTime += time_scan;
	cudaMemcpy(indices, tmp, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_bools, bools, sizeof(int) * (n), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; ++i) {
		j = tmp_bools[i] == 1 ? j + 1 : j;
	}
	cudaEventRecord(start);
	//scatter
	Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_out, dev_in, bools, indices);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totalTime += milliseconds;

	time = totalTime;
	cudaMemcpy(odata, dev_out, sizeof(int) * (n), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_in);
	cudaFree(bools);
	cudaFree(dev_out);
	cudaFree(indices);

	delete[] tmp;
	delete[] tmp_bools;
    return n - j;
}

}
}
