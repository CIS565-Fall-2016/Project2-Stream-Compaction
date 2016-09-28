#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void upSweep(int n, int d, int *data, bool isRoot) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= n) return;

	if (isRoot) {
		data[n - 1] = 0;
	}
	else {
		int prevOffset = d == 0 ? 1 : 2 << (d - 1);
		int offset = prevOffset * 2;

		if (index % offset == 0) {
			data[index + offset - 1] += data[index + prevOffset - 1];
		}
	}
}

__global__ void downSweep(int n, int d, int *data) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int prevOffset = d == 0 ? 1 : 2 << (d - 1);
	int offset = prevOffset * 2;

	if (index < n && index % offset == 0) {
		int t = data[index + prevOffset - 1];
		data[index + prevOffset - 1] = data[index + offset - 1];
		data[index + offset - 1] += t;
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
float scan(int n, int *odata, const int *idata) {
	int blockSize = 128;
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	
	int nearestPow = 2 << (ilog2ceil(n) - 1); //assume n > 0

	int* dev_data;
	cudaMalloc((void**)&dev_data, nearestPow * sizeof(int));
	checkCUDAError("cudaMalloc dev_data failed!");
	cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// Up-sweep
	int numLevels = ilog2ceil(nearestPow);
	for (int d = 0; d < numLevels; d++) {
		upSweep << <fullBlocksPerGrid, blockSize >> >(nearestPow, d, dev_data, d == (numLevels - 1));
	}

	//Down-sweep
	for (int d = numLevels; d >= 0; d--) {
		downSweep << <fullBlocksPerGrid, blockSize >> >(nearestPow, d, dev_data);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(odata, dev_data, sizeof(int) * nearestPow, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_data);
	return milliseconds;
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
int compact(int n, int *odata, const int *idata, float* timer) {
	int blockSize = 128;
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	int nearestPow = 2 << (ilog2ceil(n) - 1); //assume n > 0

	int* dev_idata;
	int* dev_odata;
	int* dev_bools;
	int* dev_indices;
	int* indices;
	
	cudaMalloc((void**)&dev_idata, nearestPow * sizeof(int));
	checkCUDAError("cudaMalloc dev_idata failed!");
	cudaMemset(dev_idata, 0, nearestPow);
	cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_odata, nearestPow * sizeof(int));
	checkCUDAError("cudaMalloc dev_odata failed!");

	cudaMalloc((void**)&dev_bools, nearestPow * sizeof(int));
	checkCUDAError("cudaMalloc dev_bools failed!");

	cudaMalloc((void**)&dev_indices, nearestPow * sizeof(int));
	checkCUDAError("cudaMalloc dev_indices failed!");
	indices = (int*)malloc(nearestPow * sizeof(int));

	cudaEvent_t start, stop;
	if (timer) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
	}

	StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(nearestPow, dev_bools, dev_idata);
	cudaMemcpy(dev_indices, dev_bools, sizeof(int) * nearestPow, cudaMemcpyDeviceToDevice);

	// Up-sweep
	int numLevels = ilog2ceil(nearestPow);
	for (int d = 0; d < numLevels; d++) {
		upSweep << <fullBlocksPerGrid, blockSize >> >(nearestPow, d, dev_indices, d == (numLevels - 1));
	}

	//Down-sweep
	for (int d = numLevels; d >= 0; d--) {
		downSweep << <fullBlocksPerGrid, blockSize >> >(nearestPow, d, dev_indices);
	}

	StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(nearestPow, dev_odata, dev_idata, dev_bools, dev_indices);
	
	if (timer) {
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		(*timer) += milliseconds;
	}

	cudaMemcpy(indices, dev_indices, sizeof(int) * nearestPow, cudaMemcpyDeviceToHost);
	int j = nearestPow - 1;
	do {
		j--;
	} while (indices[j] == indices[j + 1]);
	int compactLength = indices[j] + 1;

	cudaMemcpy(odata, dev_odata, sizeof(int) * compactLength, cudaMemcpyDeviceToHost);

	cudaFree(dev_idata);
	cudaFree(dev_odata);
	cudaFree(dev_bools);
	cudaFree(dev_indices);
	free(indices);

	return compactLength;
}

}
}
