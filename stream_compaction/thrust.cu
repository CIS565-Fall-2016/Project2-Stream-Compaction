#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
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
		thrust::device_vector<int> dv_in(n);
		thrust::device_vector<int> dv_out(n);
		thrust::copy(idata, idata + n - 1, dv_in.begin());

		#if PROFILE
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		#endif

		thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

		#if PROFILE
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Time Elapsed for trust scan (size " << n << "): " << milliseconds << std::endl;
		#endif

		thrust::copy(dv_out.begin(), dv_out.end(), odata);
	}

	void sort(int n, int *odata, const int *idata) {
		thrust::device_vector<int> dv_out(n);
		thrust::copy(idata, idata + n, dv_out.begin());

		#if PROFILE
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		#endif

		thrust::sort(dv_out.begin(), dv_out.end());

		#if PROFILE
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Time Elapsed for trust sort (size " << n << "): " << milliseconds << std::endl;
		#endif

		thrust::copy(dv_out.begin(), dv_out.end(), odata);
	}

}
}
