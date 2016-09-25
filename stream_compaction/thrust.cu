#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"
#include "profilingcommon.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

  // Convert to device vector
  thrust::device_vector<int> dev_idata(idata, idata + n);
  thrust::device_vector<int> dev_odata(odata, odata + n);

#ifdef PROFILE
  auto begin = std::chrono::high_resolution_clock::now();
  for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
#endif

  thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());

#ifdef PROFILE
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds>((end - begin) / PROFILE_ITERATIONS).count() << " ns" << std::endl;
#endif

  thrust::host_vector<int> host_odata = dev_odata;
  cudaMemcpy(odata, host_odata.data(), n * sizeof(int), cudaMemcpyHostToHost);
}

}
}
