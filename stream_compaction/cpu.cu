#include <cstdio>
#include <vector>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	int sum = 0;
	for (int i = 0; i < n; i++) {
		odata[i] = sum;
		sum += idata[i];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int j = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[j] = idata[i];
			j++;
		}
	}
	return j;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	// Map elements to boolean array
	std::vector<int> bools(n);
	for (int i = 0; i < n; i++) {
		bools[i] = (idata[i] != 0);
	}

	// Perform exclusive scan on temp array
	std::vector<int> indices(n);
	scan(n, indices.data(), bools.data());

	// Scatter 
	int elementCount;
	for (int i = 0; i < n; i++) {
		if (bools[i]) {
			odata[indices[i]] = idata[i];
			elementCount = indices[i] + 1;
		}
	}

    return elementCount;
}

}
}
