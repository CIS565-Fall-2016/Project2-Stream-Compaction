#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	if (n <= 0) {
		return;
	}
	odata[0] = idata[0];
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i];
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
	for (int i = 0; i < n; i++) {
		odata[i] = idata[i] != 0;
	}
	scan(n, odata, odata);
	int k = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			k++;
			odata[odata[i]-1] = idata[i];
		}
	}
	return k;
}

}
}
