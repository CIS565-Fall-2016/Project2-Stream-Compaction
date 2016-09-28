#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "stdio.h"
#include "stdlib.h"

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
	namespace Efficient {

		__device__ int threadIndex() {
			return (blockIdx.x * blockDim.x) + threadIdx.x;
		}

		__global__ void kernUpSweep(int n, int d, int *odata, int *idata) {
			int index = threadIndex();
			if (index >= n) return;
			int addTerm = (index + 1) % (d * 2) == 0 ? idata[index - d] : 0;
			odata[index] = idata[index] + addTerm;
		}

		__global__ void kernDownSweep(int length, int d, int *odata, int *idata) {
			int index = threadIndex();
			if (index >= length) return;

			// On the first iteration, and using only one thread, set the last element to 0.
			if ((index + 1) % d == 0) {
				int swapIndex = index - (d / 2);
				int term = (length == d) && (index == d - 1) ? 0 : idata[index];
				odata[index] = term + idata[swapIndex];
				odata[swapIndex] = term;
			}
		}

		int bufferToPow2(int n) {
			return pow(2, ceil(log2(n))); // n rounded up to the nearest power of 2 
		}

		void dev_scan(int n, int *dev_odata, int *dev_idata) {

			int bufferedLength = bufferToPow2(n);
			int numBlocks = getNumBlocks(blockSize, n); // enough blocks to allocate one thread to each array element

			// upsweep
			for (int d = 1; d <= n; d *= 2) {
				kernUpSweep << <numBlocks, blockSize >> >(n, d, dev_odata, dev_idata);

				// swap dev_idata with dev_odata
				int *swap = dev_idata;
				dev_idata = dev_odata;
				dev_odata = swap;
			}

			// downsweep
			for (int d = bufferedLength; d >= 1; d /= 2) {
				kernDownSweep << <numBlocks, blockSize >> >(bufferedLength, d, dev_odata, dev_idata);

				// swap dev_idata with dev_odata
				int *swap = dev_idata;
				dev_idata = dev_odata;
				dev_odata = swap;
			}
		}


		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata) {

			// declare arrays
			int* dev_idata;
			int* dev_odata;

			int bufferedLength = bufferToPow2(n);

			// allocate memory
			cudaMalloc((void**)&dev_idata, bufferedLength * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, bufferedLength * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			// copy memory and run the algorithm
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			dev_scan(n, dev_odata, dev_idata);

			cudaMemcpy(odata, dev_idata, n* sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
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
			// declare arrays
			int* dev_idata;
			int* dev_odata;
			int* dev_bools;
			int* dev_pingPong;
			int* dev_indices;
			int* bools = (int*)calloc(n, sizeof(int));
			int* indices = (int*)calloc(n, sizeof(int));
			int* pingPong = (int*)calloc(n, sizeof(int));

			//cudaEvent_t start, stop;
			//cudaEventCreate(&start);
			//cudaEventCreate(&stop);

			//cudaEventRecord(start);
			//saxpy << <(N + 255) / 256, 256 >> >(N, 2.0f, d_x, d_y);
			//cudaEventRecord(stop);

			//cudaEventSynchronize(stop);6
			//float milliseconds = 0;
			//cudaEventElapsedTime(&milliseconds, start, stop);

			// allocate memory
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_pingPong, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_pingPong failed!");
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_indices failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			// copy input data to device
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			////////////
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
			////////////////

			// enough blocks to allocate one thread to each array element
			int numBlocks = (n / blockSize) + 1;

			// get array of booleans determining whether 
			Common::kernMapToBoolean << <numBlocks, blockSize >> > (n, dev_bools, dev_idata);
			cudaMemcpy(dev_pingPong, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

			// allocate memory and run scan
			dev_scan(n, dev_indices, dev_pingPong);

			Common::kernScatter << <numBlocks, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

			///////////
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("%f\n", milliseconds);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			/////////


			// copy from device
			cudaMemcpy(indices, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
			int newLength = indices[n - 1] + bools[n - 1]; // return value
			cudaMemcpy(odata, dev_odata, newLength * sizeof(int), cudaMemcpyDeviceToHost);

			// free memory
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			free(indices);
			free(bools);
			free(pingPong);

			return newLength;
		}

	}
}
