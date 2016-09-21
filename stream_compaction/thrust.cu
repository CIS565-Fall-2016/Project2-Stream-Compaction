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
	int *dev_data;

	// device memory allocation
	cudaMalloc((void**)&dev_data, sizeof(int) * n);
	checkCUDAError("Failed to allocate dev_data[0]");

	// copy input data to device
	cudaMemcpy((void*)dev_data, (const void*)idata, sizeof(int) * n,
			cudaMemcpyHostToDevice);

	// cast to device ptr
	thrust::device_ptr<int> dev_thrust_data(dev_data);

	// do scan
	thrust::exclusive_scan(dev_thrust_data, dev_thrust_data + n, dev_thrust_data);

	// copy result to host
	cudaMemcpy((void*)odata, (const void*)dev_data, sizeof(int) * n,
			cudaMemcpyDeviceToHost);

	// free memory on device
	cudaFree(dev_data);
}

}
}
