#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

__global__ void kernScanStep(int n, int d, int *odata, const int *idata) {

	int s = 1 << (d - 1);
	int index = threadIdx.x + (blockIdx.x * blockDim.x) + s;
	
	if (index >= n) {
		return;
	}
	

	//if (index >= s) {
		odata[index] = idata[index] + idata[index - s];
	//}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	// create device arrays
	int *dev_in;
	int *dev_out;

	cudaMalloc((void**)&dev_in, n*sizeof(int));
	cudaMalloc((void**)&dev_out, n*sizeof(int));

	cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	scan_dev(n, dev_out, dev_in);

	cudaMemcpy(&odata[1], dev_out, (n - 1)*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_in);
	cudaFree(dev_out);
}

void scan_dev(int n, int *dev_out, int *dev_in) {
	dim3 fullBlocksPerGrid;

	for (int d = 1; d <= ilog2ceil(n); d++) {
		int s = 1 << (d - 1);
		fullBlocksPerGrid = (n - s + blockSize - 1) / blockSize;

		kernScanStep << < fullBlocksPerGrid, blockSize >> >(n, d, dev_out, dev_in);
		cudaMemcpy(dev_in, dev_out, n*sizeof(int), cudaMemcpyDeviceToDevice);
	}
}

void TestScan(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	// create device arrays
	int *dev_in;
	int *dev_out;

	cudaMalloc((void**)&dev_in, n*sizeof(int));
	cudaMalloc((void**)&dev_out, n*sizeof(int));

	cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 0; i < samp; i++) {
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		scan_dev(n, dev_out, dev_in);
		cudaThreadSynchronize(); // block until kernel is finished

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}

	cudaMemcpy(&odata[1], dev_out, (n - 1)*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_in);
	cudaFree(dev_out);

	printf("     %f\n", time );

}

}
}

