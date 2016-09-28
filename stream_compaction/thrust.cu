#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/*int *dev_idata;*/
int *dev_odata;

thrust::device_vector<int> dev_thrust_idata;
thrust::device_vector<int> dev_thrust_odata;

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	/*
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_idata failed!");*/
	cudaMalloc((void**)&dev_odata, n * sizeof(int));
	checkCUDAError("cudaMalloc dev_odata failed!");
	/*cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy idata to dev_idata failed!");*/

	thrust::device_vector<int> dev_thrust_idata(idata, idata + n);
	thrust::device_vector<int> dev_thrust_odata(odata, odata + n);

	thrust::exclusive_scan(dev_thrust_idata.begin(), dev_thrust_idata.end(),
		dev_thrust_odata.begin());

	thrust::copy(dev_thrust_odata.begin(), dev_thrust_odata.end(), dev_odata);

	cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_odata to odata failed!");
}

}
}
