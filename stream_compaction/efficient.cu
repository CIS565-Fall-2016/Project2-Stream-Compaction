#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
using namespace std::chrono;
#include <iostream>
using std::cout;
using std::endl;
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__
__global__ void upSweep(const int n, const int step, int *data) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= n) {
		return;
	}

	int rIndex = n - 1 - index;
	int mask = 1;

	for (int i = 1; i != step; (i <<= 1), (mask = mask << 1 | 1));

	if (index - step >= 0 && (rIndex & mask) == 0) {
		data[index] = data[index] + data[index - step];
	}
}

__global__ void downSweep(const int n, const int step, int *data) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= n) {
		return;
	}

	int rIndex = n - 1 - index;
	int mask = 1;

	for (int i = 1; i != step; (i <<= 1), (mask = mask << 1 | 1));

	if (index - step >= 0 && (rIndex & mask) == 0) {
		auto tmp = data[index];
		data[index] += data[index - step];
		data[index - step] = tmp;
	}
}

void scanOnGPU(const int n, int *dev_data) {
	dim3 blockCount = (n - 1) / BLOCK_SIZE + 1;
	int step;

	// up-sweep
	for (step = 1; step < n; step <<= 1) {
		upSweep<<<blockCount, BLOCK_SIZE>>>(n, step, dev_data);
	}

	// set last element to 0
	cudaMemset(&dev_data[n - 1], 0, sizeof(int));

	// down-sweep
	for (step >>= 1; step > 0; step >>= 1) {
		downSweep<<<blockCount, BLOCK_SIZE>>>(n, step, dev_data);
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    // printf("TODO\n");
	int *dev_data;

	// device memory allocation
	cudaMalloc((void**)&dev_data, sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_data");

	// copy input data to device
	cudaMemcpy((void*)dev_data, (const void*)idata, sizeof(int) * n,
			cudaMemcpyHostToDevice);

	// tic
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// do scan
	scanOnGPU(n, dev_data);

	// toc
	cudaDeviceSynchronize();
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<int, std::milli> t12 = duration_cast<duration<int, std::milli>>(t2 - t1);
	cout << "----------Time consumed: " << t12.count() << " ms----------" << endl;

	// copy result to host
	cudaMemcpy((void*)odata, (const void*)dev_data, sizeof(int) * n,
			cudaMemcpyDeviceToHost);

	// free memory on device
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
    // TODO
	int count;
	int *dev_data;
	int *dev_dataCopy;
	int *dev_bool;
	int *dev_boolScan;

	// device memory allocation
	cudaMalloc((void**)&dev_data, sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_data");

	cudaMalloc((void**)&dev_dataCopy, sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_dataCopy");

	cudaMalloc((void**)&dev_bool, sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_bool");

	cudaMalloc((void**)&dev_boolScan, sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_boolScan");

	// copy input data to device
	cudaMemcpy((void*)dev_data, (const void*)idata, sizeof(int) * n,
			cudaMemcpyHostToDevice);

	dim3 blockCount = (n - 1) / BLOCK_SIZE + 1;

	// map to booleans
	Common::kernMapToBoolean<<<blockCount, BLOCK_SIZE>>>(n, dev_bool, dev_data);

	// scan booleans
	cudaMemcpy((void*)dev_boolScan, (const void*)dev_bool, sizeof(int) * n,
			cudaMemcpyDeviceToDevice);
	scanOnGPU(n, dev_boolScan);

	// scatter
	cudaMemcpy((void*)dev_dataCopy, (const void*)dev_data, sizeof(int) * n,
			cudaMemcpyDeviceToDevice);
	Common::kernScatter<<<blockCount, BLOCK_SIZE>>>(n, dev_data, dev_dataCopy,
			dev_bool, dev_boolScan);

	// copy result to host
	cudaMemcpy((void*)odata, (const void*)dev_data, sizeof(int) * n,
			cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)&count, (const void*)&dev_boolScan[n - 1], sizeof(int),
			cudaMemcpyDeviceToHost);

	// free memory on device
	cudaFree(dev_data);
	cudaFree(dev_dataCopy);
	cudaFree(dev_bool);
	cudaFree(dev_boolScan);

    return count;
}

}
}
