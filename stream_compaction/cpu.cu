#include <cstdio>
#include "common.h"
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	for (int i = 0; i < n-1; i++) {
		odata[i + 1] = odata[i] + idata[i];
	}
}

/**
* CPU scatter.
*/
void scatter(int n, int *odata,
	const int *idata, const int *bools, const int *indices) {

	for (int i = 0; i < n - 1; i++) {
		if (bools[i] == 1) odata[indices[i]] = idata[i];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int r = 0;

	for (int i = 0; i < n-1; i++){
		if (idata[i] != 0) {
			odata[r++] = idata[i];
		}
	}

    return r;
}

void printArray(int n, int *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%3d ", a[i]);
	}
	printf("]\n");
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(const int n, int *odata, const int *idata) {

	// create arrays
	int *indices = new int[n];
	int *bools = new int[n];
	int rtn = -1;
	indices[0] = 0;

	for (int i = 0; i < n; i++) {
		bools[i] = !(idata[i] == 0);
	}

	scan(n, indices, bools);
	scatter(n, odata, idata, bools, indices);
	rtn = indices[n - 1];

	delete indices;
	delete bools;

	return rtn;
}

void TestScan(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	for (int i = 0; i < samp; i++) {
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		scan(n, odata, idata);

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}
	printf("     %f\n", time);

}

void TestCompact(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	for (int i = 0; i < samp; i++) {
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		compactWithScan(n, odata, idata);

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}
	printf("     %f\n", time);

}

void TestCompactWithoutScan(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	for (int i = 0; i < samp; i++) {
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		compactWithoutScan(n, odata, idata);

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}
	printf("     %f\n", time);

}

}
}
