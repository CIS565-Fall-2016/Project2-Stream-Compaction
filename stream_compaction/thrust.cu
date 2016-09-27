#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	thrust::host_vector<int> host_odata(n);
	thrust::device_vector<int> dev_thrust_odata = host_odata;

	int *dev_idata;
	cudaMalloc((void**)&dev_idata, sizeof(int)*n);
	checkCUDAErrorFn("Failed to allocate dev_data");
	cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
	checkCUDAErrorFn("Failed to copy dev_data");

	// Use a thrust pointer because the vector wasn't working for me
	thrust::device_ptr<int> dev_thrust_idata(dev_idata);

	// Scan and copy back
	thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata.begin());
	thrust::copy(dev_thrust_odata.begin(), dev_thrust_odata.end(), odata);
	
	// Free mem
	cudaFree(dev_idata);
}

}
}
