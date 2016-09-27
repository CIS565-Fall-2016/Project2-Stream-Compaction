#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128
namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void kernUpSweep(int n, int offset, int *buf) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	int idx = (index + 1) * (offset * 2) - 1;
	if (idx >= n) return;
	//if ((index + 1) % (offset * 2) == 0) return;
	
	buf[idx] += buf[idx - offset];
	//buf[index] += buf[index - offset];
}

__global__ void kernDownSweep(int n, int offset, int *buf) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	int idx = (index + 1) * (offset * 2) - 1;
	if (idx >= n) return;

	int t = buf[idx - offset];
	buf[idx - offset] = buf[idx];
	buf[idx] += t;
}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	int *buf;
	int padded = 1 << ilog2ceil(n);

	cudaMalloc((void**)&buf, padded * sizeof(int));
	checkCUDAError("cudaMalloc buf failed!");

	cudaMemcpy(buf, idata, padded * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	int offset;
	for (int i = 0; i <= ilog2(padded); i++) {
		kernUpSweep << <fullBlocksPerGrid, blockSize >> >(padded, 1 << i, buf);
	}

	cudaMemset(buf + padded - 1, 0, sizeof(int));
	for (int i = ilog2(padded); i >= 0; i--) {
		kernDownSweep << <fullBlocksPerGrid, blockSize >> >(padded, 1 << i, buf);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, end);
	printf("Work-Efficient scan: %f ms\n", milliseconds);

	cudaMemcpy(odata, buf, padded * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(buf);
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
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	int *bools, *indices, *in, *out;
	
	cudaMalloc((void**)&bools, n * sizeof(int));
	cudaMalloc((void**)&indices, n * sizeof(int));
	cudaMalloc((void**)&in, n * sizeof(int));
	cudaMalloc((void**)&out, n * sizeof(int));

	cudaMemcpy(in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

	float total = 0;
	float milliseconds = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, bools, in);
	
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
	total += milliseconds;
	
	cudaMemcpy(odata, bools, n * sizeof(int), cudaMemcpyDeviceToHost);
	scan(n, odata, odata);
	int lenCompacted = odata[n - 1];
	cudaMemcpy(indices, odata, n * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, out, in, bools, indices);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
	total += milliseconds;
	printf("Work-Efficient Compact: %f ms\n", total);
	cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(bools);
	cudaFree(indices);
	cudaFree(in);
	cudaFree(out);

	return lenCompacted;
}

}
}
