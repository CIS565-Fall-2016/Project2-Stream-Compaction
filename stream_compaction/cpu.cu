#include <cstdio>
#include "cpu.h"
#include "timer.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	if (n <= 0) {
		return;
	}
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}

}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int k = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i]) {
			odata[k++] = idata[i];
		}
	}
	return k;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *scanResults = new int[n];

	// mapping boolean function
	for (int i = 0; i < n; i++) {
		odata[i] = idata[i] != 0;
	}

	//scan
	scan(n, scanResults, odata);
	
	//compaction
	int k = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i]) {
			k++;
			odata[scanResults[i]] = idata[i];
		}
	}
	return k;
}

}
}
