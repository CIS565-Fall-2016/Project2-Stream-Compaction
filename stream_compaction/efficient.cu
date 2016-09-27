#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {
	__global__ void kernScanUp(int n, int pow_2_d, int pow_2_d_one, int*g_idata) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Calculate children indices
		int ai = pow_2_d_one * (index + 1) - pow_2_d - 1;
		int bi = pow_2_d_one * (index + 1) - 1;

		if (ai >= n || bi >= n) {
			return;
		}

		g_idata[bi] += g_idata[ai];
	}

	__global__ void kernZeroLastValue(int n, int *data) {
		data[n - 1] = 0;
	}

	__global__ void kernScanDown(int n, int pow_2_d, int pow_2_d_one, int *g_idata) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		int ai = pow_2_d_one * (index + 1) - pow_2_d - 1;
		int bi = pow_2_d_one * (index + 1) - 1;

		float t = g_idata[ai];
		g_idata[ai] = g_idata[bi];
		g_idata[bi] += t;
	}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid;
	dim3 threadsPerBlock(128);

	// Allocate GPU space
	int* dev_in;

	cudaMalloc((void**)&dev_in, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_in.");

	// Copy array to device
	cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	
	// Scan up
	for (int d = 0; d < ilog2ceil(n); d++) {

		int n2 = pow(2, ilog2ceil(n) - d - 1);
		fullBlocksPerGrid = dim3((n2 + 128 - 1) / 128);

		kernScanUp << <fullBlocksPerGrid, threadsPerBlock >> >(n, pow(2, d), pow(2, d + 1), dev_in);

	}

	// Make last value 0
	kernZeroLastValue<<<1, 1>>>(n, dev_in);

	// Scan down
	for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
		int n2 = pow(2, ilog2ceil(n) - d - 1);
		fullBlocksPerGrid = dim3((n2 + 128 - 1) / 128);

		kernScanDown << <fullBlocksPerGrid, threadsPerBlock >> >(n, pow(2, d), pow(2, d + 1), dev_in);
	}
	
	// Copy data back to host
	cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy back failed!");

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
int compact(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + 128 - 1) / 128);
	dim3 threadsPerBlock(128);

	// Allocate GPU space
	int* dev_in;
	int* dev_out;
	int* dev_map;
	int* dev_scan;

	cudaMalloc((void**)&dev_in, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_in.");

	cudaMalloc((void**)&dev_out, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_in.");

	cudaMalloc((void**)&dev_map, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_map.");

	cudaMalloc((void**)&dev_scan, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_map.");

	// Copy array to device
	cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

	// Map 1s and 0s
	StreamCompaction::Common::kernMapToBoolean<< < fullBlocksPerGrid, threadsPerBlock>>>(n, dev_map, dev_in);

	// Scan the map
	scan(n, dev_scan, dev_map);

	// Scatter
	StreamCompaction::Common::kernScatter<< <fullBlocksPerGrid, threadsPerBlock >> >(n, dev_out, dev_in, dev_map, dev_scan);

	// Copy scan data back to host
	// This feels kind of hacky, but I don't know of a better way to acess device arrays from the host

	cudaMemcpy(odata, dev_scan, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy back failed!");

	int ret = odata[n - 1];

	// Copy data back to host
	cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy back failed!");

	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_scan);
	cudaFree(dev_map);

    return ret;
}

}
}
