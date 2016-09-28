#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"
#include <chrono>

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
double scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	thrust::host_vector<int> hv_in(n);

	for (int i = 0; i < n; ++i)
	{
		hv_in[i] = idata[i];
	}

	thrust::device_vector<int> dv_in(hv_in), dv_out(n);

	// record time
	auto start = std::chrono::system_clock::now();

	thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

	std::chrono::duration<double, std::milli> diff = (std::chrono::system_clock::now() - start);
	//printf("Thrust scan took %fms\n", diff.count());

	thrust::copy(dv_out.begin(), dv_out.end(), odata);

	return diff.count();
}

}
}
