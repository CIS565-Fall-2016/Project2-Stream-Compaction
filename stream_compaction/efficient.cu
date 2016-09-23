#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

	__global__ void kernScanUp(int n, int d, int step, int *g_odata, int*g_idata) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index < d) {
			int ai = step * (2 * index + 1) - 1;
			int bi = step * (2 * index + 2) - 1;

			g_idata[bi] += g_idata[ai];
		}
	}

	__global__ void kernScanDown(int n, int d, int step, int *g_odata, int *g_idata) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index == 0) {
			g_idata[n - 1] = 0;
		}

		// traverse down tree & build scan  
		if (index < d) {
			int ai = step * (2 * index + 1) - 1;
			int bi = step * (2 * index + 2) - 1;


			float t = g_idata[ai];
			g_idata[ai] = g_idata[bi];
			g_idata[bi] += t;
		}
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + 128 - 1) / 128);
	dim3 threadsPerBlock(128);

	int* dev_in;
	int* dev_out;

	cudaMalloc((void**)&dev_in, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_in.");

	cudaMalloc((void**)&dev_out, n * sizeof(int));
	checkCUDAError("cudaMalloc Error dev_out.");

	cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	
	for (int d = 0; d < ilog2ceil(n); d++) {
		kernScanUp << <fullBlocksPerGrid, threadsPerBlock >> >(n, d, pow(2, d+1), dev_out, dev_in);
	}

	for (int d = ilog2ceil(n); d >= 0; d--) {
		kernScanDown << <fullBlocksPerGrid, threadsPerBlock >> >(n, d, pow(2, d + 1), dev_out, dev_in);
	}
		
	cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy back failed!");

	cudaFree(dev_in);
	cudaFree(dev_out);
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
    return -1;
}

}
}
