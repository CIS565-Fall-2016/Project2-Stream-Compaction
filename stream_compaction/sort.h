#pragma once

namespace StreamCompaction {
	namespace Sort {
		void radix(int n, const int k, int *odata, const int *idata);
		void radix_dev(int n, const int k, int *dev_out, int *dev_in);

		void TestSort(int n, const int k, int *odata, const int *idata);
	}
}
