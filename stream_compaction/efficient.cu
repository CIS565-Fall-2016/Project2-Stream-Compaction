#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void kernUpStep(int n, int d, int *data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}

	int s = pow((double)2, (double)(d + 1));

	if (fmod((double) index, (double)s) == 0) {
		data[index + s - 1] += data[index + s / 2 - 1];
	}
}

__global__ void kernDownStep(int n, int d, int *data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}

	int s = pow((double)2, (double)(d + 1));

	if (fmod((double) index, (double)s) == 0) {
		int t = data[index + s / 2 - 1];
		data[index + s / 2 - 1] = data[index + s - 1];
		data[index + s - 1] += t;
	}
}

/**
* Performs prefix-sum (aka scan) on idata, storing the result into odata.
*/
void scan(int n, int *odata, const int *idata) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	// create device arrays
	int *dev_out;

	cudaMalloc((void**)&dev_out, n*sizeof(int));
	cudaMemcpy(dev_out, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	int d = 0;
	for (d; d < ilog2ceil(n); d++) {
		kernUpStep << < fullBlocksPerGrid, blockSize >> >(n, d, dev_out);
	}

	cudaMemset(&dev_out[n - 1], 0, sizeof(int));
	for (d; d >= 0; d--) {
		kernDownStep << < fullBlocksPerGrid, blockSize >> >(n, d, dev_out);
	}

	cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_out);
}

/**
* Performs prefix-sum (aka scan) on idata, storing the result into odata.
* For use with arrays intiialized on GPU already.
*/
void scan_dev(int n, int *dev_data) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	// create device arrays
	int d = 0;
	for (d; d < ilog2ceil(n); d++) {
		kernUpStep << < fullBlocksPerGrid, blockSize >> >(n, d, dev_data);
	}

	cudaMemset(&dev_data[n - 1], 0, sizeof(int));
	for (d; d >= 0; d--) {
		kernDownStep << < fullBlocksPerGrid, blockSize >> >(n, d, dev_data);
	}

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

	// create device arrays
	int *dev_out;
	int *dev_in;
	int *dev_indices;
	int *dev_bools;
	int rtn = -1;

	cudaMalloc((void**)&dev_out, n*sizeof(int));
	cudaMalloc((void**)&dev_in, n*sizeof(int));
	cudaMalloc((void**)&dev_indices, n*sizeof(int));
	cudaMalloc((void**)&dev_bools, n*sizeof(int));


	cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);
	StreamCompaction::Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> >(n, dev_bools, dev_in);

	// scan without wasteful device-host-device write
	cudaMemcpy(dev_indices, dev_bools, n*sizeof(int), cudaMemcpyDeviceToDevice);
	scan_dev(n, dev_indices);

	// scatter
	StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, dev_bools, dev_indices);

	cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&rtn, &dev_indices[n-1], sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(dev_out);
	cudaFree(dev_in);
	cudaFree(dev_bools);
	cudaFree(dev_indices);

    return rtn;
}

}
}
