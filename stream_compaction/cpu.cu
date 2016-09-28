#include <cstdio>
#include "cpu.h"
#include "common.h"
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
	namespace CPU {

		/**
		* CPU scan (prefix sum).
		*/
		void scan(int n, int *odata, const int *idata) {
			// TODO
			int valToWrite = 0;
			int sumOfPrev2;
			for (int i = 0; i < n; i++) {
				sumOfPrev2 = valToWrite + idata[i];
				odata[i] = valToWrite;
				valToWrite = sumOfPrev2;
			}
		}

		/**
		* CPU stream compaction without using the scan function.
		*
		* @returns the number of elements remaining after compaction.
		*/
		int compactWithoutScan(int n, int *odata, const int *idata) {
			// TODO
			int j = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[j++] = idata[i];
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
			// TODO

			for (int i = 0; i < n; i++) {
				odata[i] = idata[i] != 0;
			}

			scan(n, odata, odata);


			int retVal = odata[n - 1] + (idata[n - 1] != 0);
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[odata[i]] = idata[i];
				}
			}
			return retVal;
		}
	}
}
