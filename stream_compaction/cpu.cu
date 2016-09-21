#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    //printf("TODO\n");
	odata[0] = 0;
	for (int i = 1; i < n; ++i) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
	int m = 0;

	for (int i = 0; i < n; ++i) {
		if (idata[i] == 0) continue;
		odata[m++] = idata[i];
	}

    return m;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
	int *nonZero = new int [n];
	int m = 0;

	for (int i = 0; i < n; ++i) {
		nonZero[i] = idata[i] == 0 ? 0 : 1;
	}

	scan(n, odata, nonZero);
	m = odata[n - 1];

	for (int i = 0; i < n; ++i) {
		if (nonZero[i] == 0) continue;
		odata[odata[i]] = idata[i];
	}

	delete [] nonZero;

    return m;
}

}
}
