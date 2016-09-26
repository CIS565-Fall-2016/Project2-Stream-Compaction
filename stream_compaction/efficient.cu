#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Efficient {

		// TODO: __global__
		__global__ void kernScanEfficient(int N, int interval, int *data)
		{
			// up sweep
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) return;
			int cur_index = index + 2 * interval - 1;
			int last_index = index + interval - 1;
			if (cur_index >= N) return;

			data[cur_index] = data[last_index];
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			// TODO
			printf("TODO\n");
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
			// TODO
			return -1;
		}

	}
}
