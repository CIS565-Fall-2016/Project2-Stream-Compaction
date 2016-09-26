#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

void printArray(int n, int *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%3d ", a[i]);
	}
	printf("]\n");
}

__global__ void kernUpStep(int n, int d, int *data) {

	int s = 1 << (d + 1);
	int index = (threadIdx.x + (blockIdx.x * blockDim.x)) *s;

	if (index >= n) {
		return;
	}
	
	//if (fmod((double) index+1, (double)s) == 0) {
		data[index + s - 1] += data[index + s / 2 - 1];
	//}
}

__global__ void kernDownStep(int n, int d, int *data) {
	
	int s = 1 << (d + 1);
	int index = (threadIdx.x + (blockIdx.x * blockDim.x)) * s;
	
	if (index >= n) {
		return;
	}


	//if (fmod((double) index, (double)s) == 0) {
		int t = data[index + s / 2 - 1];
		data[index + s / 2 - 1] = data[index + s - 1];
		data[index + s - 1] += t;
	//}
}

/**
* Performs prefix-sum (aka scan) on idata, storing the result into odata.
*/
void scan(int n, int *odata, const int *idata) {
	// create device arrays
	int *dev_out;

	cudaMalloc((void**)&dev_out, n*sizeof(int));
	cudaMemcpy(dev_out, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	scan_dev(n, dev_out);

	cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_out);
}

/**
* Performs prefix-sum (aka scan) on idata, storing the result into odata.
* For use with arrays intiialized on GPU already.
*/
void scan_dev(int n, int *dev_in) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	// create device arrays to pad to power of 2 size array
	//int pot = pow(2, ilog2ceil(n));
	int pot = 1 << ilog2ceil(n);
	int *dev_data;
	cudaMalloc((void**)&dev_data, pot*sizeof(int));
	cudaMemset(dev_data, 0, pot*sizeof(int));
	cudaMemcpy(dev_data, dev_in, n*sizeof(int), cudaMemcpyDeviceToDevice);


	int d = 0;
	for (d; d < ilog2ceil(pot); d++) {
		fullBlocksPerGrid.x = ((pot / pow(2, d+1) + blockSize - 1) / blockSize);
		kernUpStep << < fullBlocksPerGrid, blockSize >> >(pot, d, dev_data);
	}

	cudaMemset(&dev_data[pot - 1], 0, sizeof(int));
	for (d = ilog2ceil(pot); d >= 0; d--) {
		fullBlocksPerGrid.x = ((pot / pow(2, d+1) + blockSize - 1) / blockSize);
		kernDownStep << < fullBlocksPerGrid, blockSize >> >(pot, d, dev_data);
	}


	cudaMemcpy(dev_in, dev_data, n*sizeof(int), cudaMemcpyDeviceToDevice);
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


int compact_dev(int n, int *dev_out, const int *dev_in) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	// create device arrays
	int *dev_indices;
	int *dev_bools;
	int rtn = -1;

	cudaMalloc((void**)&dev_indices, n*sizeof(int));
	cudaMalloc((void**)&dev_bools, n*sizeof(int));

	StreamCompaction::Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> >(n, dev_bools, dev_in);

	// scan without wasteful device-host-device write
	cudaMemcpy(dev_indices, dev_bools, n*sizeof(int), cudaMemcpyDeviceToDevice);
	scan_dev(n, dev_indices);

	// scatter
	StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, blockSize >> >(n, dev_out, dev_in, dev_bools, dev_indices);

	cudaMemcpy(&rtn, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_bools);
	cudaFree(dev_indices);

	return rtn;
}


void TestScan(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	int *dev_out;

	cudaMalloc((void**)&dev_out, n*sizeof(int));
	cudaMemcpy(dev_out, idata, n*sizeof(int), cudaMemcpyHostToDevice);


	for (int i = 0; i < samp; i++) {
		cudaMemcpy(dev_out, idata, n*sizeof(int), cudaMemcpyHostToDevice);
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		scan_dev(n, dev_out);

		cudaThreadSynchronize(); // block until kernel is finished

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}

	cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_out);
	printf("     %f\n", time);

}

void TestCompact(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	int *dev_out;
	int *dev_in;
	cudaMalloc((void**)&dev_out, n*sizeof(int));
	cudaMalloc((void**)&dev_in, n*sizeof(int));

	for (int i = 0; i < samp; i++) {
		cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);

		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		compact_dev(n, odata, idata);
		cudaThreadSynchronize(); // block until kernel is finished

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}

	cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_out);
	cudaFree(dev_in);

	printf("     %f\n", time);

}

}
}
