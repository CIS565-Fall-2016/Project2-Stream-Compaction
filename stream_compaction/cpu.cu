#include <cstdio>
#include <cstdlib>
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
	int nonZeroCount = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[nonZeroCount] = idata[i];
			nonZeroCount++;
		}
	}
	return nonZeroCount;
}

void scatter(int n, int * odata, const int * idata, const int * scatterTargets) {
	for (int i = 0; i < n; i++) {
		odata[scatterTargets[i]] = idata[i];
	}
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int * nonZeroElements = (int *)malloc(n * sizeof(int));
	int * scanCounts = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		nonZeroElements[i] = (idata[i] == 0) ? 0 : 1;
	}
	scan(n, scanCounts, nonZeroElements);
	scatter(n, odata, idata, scanCounts);
	int remainingCount = nonZeroElements[n - 1] + scanCounts[n - 1];
	free(nonZeroElements);
	free(scanCounts);
	return remainingCount;
}

}
}
