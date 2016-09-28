#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
using namespace std::chrono;
#include <iostream>
using std::cout;
using std::endl;
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__
__global__ void accumulate(const int n, const int step, int *idata, int *odata) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= n) {
		return;
	}

	if (index - step < 0) {
		odata[index] = idata[index];
	} else {
		odata[index] = idata[index] + idata[index - step];
	}
}

__global__ void shiftRight(const int n, int *idata, int *odata) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= n) {
		return;
	} else if (index == 0) {
		odata[index] = 0;
		return;
	}

	odata[index] = idata[index - 1];
}

// swap
inline void swap(int &x, int &y) {
	auto tmp = x; x = y; y = tmp;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    // printf("TODO\n");
	int *dev_data[2];
	int input = 1;
	int output = 0;

	// device memory allocation
	cudaMalloc((void**)&dev_data[0], sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_data[0]");

	cudaMalloc((void**)&dev_data[1], sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_data[1]");

	// copy input data to device
	cudaMemcpy((void*)dev_data[input], (const void*)idata, sizeof(int) * n,
			cudaMemcpyHostToDevice);

	// tic
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// do scan
	dim3 blockCount = (n - 1) / BLOCK_SIZE + 1;

	shiftRight<<<blockCount, BLOCK_SIZE>>>(n, dev_data[input], dev_data[output]);

	for (int step = 1; step < n; step <<= 1) {
		swap(input, output);
		accumulate<<<blockCount, BLOCK_SIZE>>>(n, step, dev_data[input], dev_data[output]);
	}

	// toc
	cudaDeviceSynchronize();
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<int, std::milli> t12 = duration_cast<duration<int, std::milli>>(t2 - t1);
	cout << "----------Time consumed: " << t12.count() << " ms----------" << endl;

	// copy result to host
	cudaMemcpy((void*)odata, (const void*)dev_data[output], sizeof(int) * n,
			cudaMemcpyDeviceToHost);

	// free memory on device
	cudaFree(dev_data[0]);
	cudaFree(dev_data[1]);
}

}
}
