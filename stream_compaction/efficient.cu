#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
namespace Efficient {

__global__ void kernScanUpsweep(int n, int d, int * data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= (n >> d)) {
		return;
	}
	int k = index << d;
	data[k + (1 << d) - 1] += data[k + (1 << (d - 1)) - 1];
}

__global__ void kernScanDownsweep(int n, int d, int * data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= (n >> d)) {
		return;
	}
	int k = index << d;
	int t = data[k + (1 << d) - 1];
	data[k + (1 << d) - 1] += data[k + (1 << (d - 1)) - 1];
	data[k + (1 << (d - 1)) - 1] = t;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	int * dev_data;
	int logCeil = ilog2ceil(n);
	int nCeil = 1 << logCeil;

	cudaMalloc((void**)&dev_data, nCeil * sizeof(int));
	cudaMemset((void*)dev_data, 0, nCeil * sizeof(int));
	cudaMemcpy((void*)dev_data, (void*)idata, n * sizeof(int), cudaMemcpyHostToDevice);

#if TIMING == 1
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif

	for (int i = 1; i <= logCeil; i++) {
		int gridSize = ((nCeil >> i) + blockSize - 1) / blockSize;
		kernScanUpsweep << <gridSize, blockSize >> >(nCeil, i, dev_data);
	}

	cudaMemset((void*)&dev_data[nCeil - 1], 0, sizeof(int));

	for (int i = logCeil; i >= 1; i--) {
		int gridSize = ((nCeil >> i) + blockSize - 1) / blockSize;
		kernScanDownsweep << <gridSize, blockSize >> >(nCeil, i, dev_data);
	}

#if TIMING == 1
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Efficient scan: %f milliseconds\n", milliseconds);
#endif

	cudaMemcpy((void*)odata, (void*)dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_data);
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
int compact(int n, int *odata, const int *idata) {
	int * dev_bools;
	int * dev_idata;
	int * dev_odata;
	int * dev_indices;
	cudaMalloc((void**)&dev_bools, n * sizeof(int));
	cudaMalloc((void**)&dev_indices, n * sizeof(int));
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	cudaMalloc((void**)&dev_odata, n * sizeof(int));

	// Map to booleans
	cudaMemcpy((void*)dev_idata, (void*)idata, n * sizeof(int), cudaMemcpyHostToDevice);
	StreamCompaction::Common::kernMapToBoolean << <n + blockSize - 1, blockSize >> >(n, dev_bools, dev_idata);
	int * temp = (int *)malloc(n * sizeof(int));
	cudaMemcpy((void*)temp, (void*)dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

	// Do exclusive scan
	scan(n, temp, temp);
	int compactedCount = temp[n - 1] + ((idata[n - 1] == 0) ? 0 : 1);

	// Scatter
	cudaMemcpy((void*)dev_indices, (void*)temp, n * sizeof(int), cudaMemcpyHostToDevice);
	StreamCompaction::Common::kernScatter << <n + blockSize - 1, blockSize >> >(n, dev_odata, dev_idata, dev_bools, dev_indices);
	cudaMemcpy((void*)odata, (void*)dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

	free(temp);
	cudaFree(dev_bools);
	cudaFree(dev_idata);
	cudaFree(dev_odata);
	cudaFree(dev_indices);

    return compactedCount;
}

}
}
