#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#define MEASURE_EXEC_TIME
#include "thrust.h"

namespace StreamCompaction {
	namespace Thrust {

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
#ifdef MEASURE_EXEC_TIME
		float scan(int n, int *odata, const int *idata)
		{
			if (n <= 0 || !odata || !idata || odata == idata)
			{
				return -1;
			}
#else
		void scan(int n, int *odata, const int *idata)
		{
			if (n <= 0 || !odata || !idata || odata == idata)
			{
				return;
			}
#endif
			// TODO use `thrust::exclusive_scan`
			// example: for device_vectors dv_in and dv_out:
			// thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			const size_t kArraySizeInByte = n * sizeof(int);
			int *idata_dev = nullptr, *odata_dev = nullptr;

			cudaMalloc(&idata_dev, kArraySizeInByte);
			cudaMalloc(&odata_dev, kArraySizeInByte);
			cudaMemcpy(idata_dev, idata, kArraySizeInByte, cudaMemcpyHostToDevice);

			thrust::device_ptr<int> thrust_idata(idata_dev);
			thrust::device_ptr<int> thrust_odata(odata_dev);

#ifdef MEASURE_EXEC_TIME
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
#endif

			thrust::exclusive_scan(thrust_idata, thrust_idata + n, thrust_odata);

#ifdef MEASURE_EXEC_TIME
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float millisceconds = 0;
			cudaEventElapsedTime(&millisceconds, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
#endif

			cudaMemcpy(odata, odata_dev, kArraySizeInByte, cudaMemcpyDeviceToHost);
			cudaFree(idata_dev);
			cudaFree(odata_dev);
			cudaDeviceSynchronize(); // make sure result is ready

#ifdef MEASURE_EXEC_TIME
			return millisceconds;
#endif
		}

	}
}
