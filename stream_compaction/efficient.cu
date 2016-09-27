#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blockSize 128

namespace StreamCompaction {
	namespace Efficient {

		__global__ void kernUpSweep(int n, int *data, int d){
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			if ((index / float(d) - index / d) > 0)  return;
			data[index + d - 1] += data[index + d / 2 - 1];
		}

		__global__ void kernDownSweep(int n, int *data, int d){
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			if ((index / float(d) - index / d) > 0)  return;
			int t = data[index + d / 2 - 1];
			data[index + d / 2 - 1] = data[index + d - 1];
			data[index + d - 1] += t;
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {

			// Pad and resize idata into temporary array if it is not a power of 2
			int powerOf2Size = std::ceil(std::log2(n));
			int newN = std::pow(2, powerOf2Size);
			int *temp = new int[newN];
			for (int x = 0; x < n; x++){
				temp[x] = idata[x];
			}

			dim3 fullBlocksPerGrid((newN + blockSize - 1) / blockSize);

			// Create GPU array pointers
			int *dev_data;

			// Allocate GPU space
			cudaMalloc((void**)&dev_data, newN * sizeof(int));
			checkCUDAErrorFn("Failed to allocate dev_data");

			cudaMemcpy(dev_data, temp, sizeof(int)*newN, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Failed to copy dev_iData");

			//cudaEvent_t start, stop;
			//cudaEventCreate(&start);
			//cudaEventCreate(&stop);
			//cudaEventRecord(start);
			// Perform scan
			for (int x = 1; x < newN; x *= 2) {
				kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(newN, dev_data, 2 * x);
			}
			cudaMemcpy(temp, dev_data, sizeof(int)*newN, cudaMemcpyDeviceToHost);
			temp[newN - 1] = 0;
			cudaMemcpy(dev_data, temp, sizeof(int)*newN, cudaMemcpyHostToDevice);
			for (int x = newN / 2; x > 0; x /= 2) {
				kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(newN, dev_data, 2 * x);
			}

			//cudaEventRecord(stop);

			//cudaEventSynchronize(stop);
			//float milliseconds = 0;
			//cudaEventElapsedTime(&milliseconds, start, stop);
			//std::cout << milliseconds << std::endl;

			cudaMemcpy(temp, dev_data, sizeof(int)*newN, cudaMemcpyDeviceToHost);
			for (int x = 0; x < n; x++){
				odata[x] = temp[x];
			}

			cudaFree(dev_data);
			delete[] temp;
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

			int *dev_bools;
			int *dev_idata;
			int *dev_odata;
			int *dev_indices;

			// Allocate GPU space
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("Failed to allocate dev_data");

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("Failed to allocate dev_data");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("Failed to allocate dev_data");

			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorFn("Failed to allocate dev_data");

			cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Failed to copy dev_iData");

			//cudaEvent_t start, stop;
			//cudaEventCreate(&start);
			//cudaEventCreate(&stop);
			//cudaEventRecord(start);
			
			Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);

			int *bools = new int[n];

			cudaMemcpy(bools, dev_bools, sizeof(int)*n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Failed to copy bools");

			scan(n, odata, bools);

			// Find number of elements. It is the last value in the indices array. If the 
			// last entry of bool is 1, that means we need to add 1 since the value be an index,
			// not the count of elements.
			int numberOfElements = odata[n - 1];
			if (bools[n - 1] == 1) numberOfElements++;

			// Copy indices over
			cudaMemcpy(dev_indices, odata, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Failed to copy dev_oData");

			Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

			//cudaEventRecord(stop);

			//cudaEventSynchronize(stop);
			//float milliseconds = 0;
			//cudaEventElapsedTime(&milliseconds, start, stop);
			//std::cout << milliseconds << std::endl;

			// Bring odata back
			cudaMemcpy(odata, dev_odata, sizeof(int)*n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Failed to copy dev_oData");

			// Free memories
			cudaFree(dev_bools);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_indices);
			delete[] bools;

			return numberOfElements;
		}

	}
}
