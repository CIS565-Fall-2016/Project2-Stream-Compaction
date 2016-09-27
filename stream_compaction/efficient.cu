#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
namespace Efficient {

	__global__ void upSweep(int n, int *idata, int d) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		if (k % (d * 2) == (d * 2) - 1) {
			idata[k] += idata[k - d];
		}

	}

	__global__ void downSweep(int n, int *idata, int d) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		int temp;
		if (k % (d * 2) == (d * 2) - 1) {
			//printf("kernel: %d", k);
			temp = idata[k - d];
			idata[k - d] = idata[k];  // Set left child to this node’s value
			idata[k] += temp;
		}

	}

	__global__ void makeElementZero(int *data, int index) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index == k) {
			data[k] = 0;
		}
	}

	__global__ void copyElements(int n, int *src, int *dest) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		dest[index] = src[index];
	}

	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int *odata, const int *idata) {
		int *dev_idata;

		int paddedArraySize = 1 << ilog2ceil(n);

		dim3 fullBlocksPerGrid((paddedArraySize + blockSize - 1) / blockSize);

		cudaMalloc((void**)&dev_idata, paddedArraySize * sizeof(int));
		checkCUDAError("Cannot allocate memory for idata");

		cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);

		#if PROFILE
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		#endif

		for (int d = 0; d < ilog2ceil(paddedArraySize); d++) {
			upSweep << <fullBlocksPerGrid, blockSize >> >(paddedArraySize, dev_idata, 1<<d);
		}

		makeElementZero << <fullBlocksPerGrid, blockSize >> >(dev_idata, paddedArraySize - 1);

		for (int d = ilog2ceil(paddedArraySize) - 1; d >= 0; d--) {
			downSweep << <fullBlocksPerGrid, blockSize >> >(paddedArraySize, dev_idata, 1<<d);
		}

		#if PROFILE
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Time Elapsed for Efficient scan (size " << n << "): " << milliseconds << std::endl;
		#endif
	
		cudaMemcpy(odata, dev_idata, n*sizeof(int), cudaMemcpyDeviceToHost);
	
		cudaFree(dev_idata);
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
		int *dev_idata;
		int *dev_boolean;
		int *dev_odata;
		int *dev_indices;
		int count;

		int paddedArraySize = 1 << ilog2ceil(n);

		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGridPadded((paddedArraySize + blockSize - 1) / blockSize);

		cudaMalloc((void**)&dev_idata, n * sizeof(int));
		checkCUDAError("Cannot allocate memory for idata");
		cudaMalloc((void**)&dev_odata, n * sizeof(int));
		checkCUDAError("Cannot allocate memory for odata");
		cudaMalloc((void**)&dev_boolean, paddedArraySize * sizeof(int));
		checkCUDAError("Cannot allocate memory for boolean");
		cudaMalloc((void**)&dev_indices, paddedArraySize * sizeof(int));
		checkCUDAError("Cannot allocate memory for dev_indices");

		cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);

		#if PROFILE
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		#endif

		StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_boolean, dev_idata);

		copyElements << <fullBlocksPerGrid, blockSize >> >(n, dev_boolean, dev_indices);

		for (int d = 0; d < ilog2ceil(paddedArraySize); d++) {
			upSweep << <fullBlocksPerGridPadded, blockSize >> >(paddedArraySize, dev_indices, 1 << d);
		}

		makeElementZero << <fullBlocksPerGridPadded, blockSize >> >(dev_indices, paddedArraySize - 1);

		for (int d = ilog2ceil(paddedArraySize) - 1; d >= 0; d--) {
			downSweep << <fullBlocksPerGrid, blockSize >> >(paddedArraySize, dev_indices, 1 << d);
		}

		StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_odata, dev_idata, dev_boolean, dev_indices);

		#if PROFILE
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Time Elapsed for Efficient compact (size " << n << "): " << milliseconds << std::endl;
		#endif

		cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&count, dev_indices + paddedArraySize - 1, sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_idata);
		cudaFree(dev_odata);
		cudaFree(dev_boolean);
		cudaFree(dev_indices);
		return count;
	}

}
}
