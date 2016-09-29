#pragma once

namespace StreamCompaction {
namespace Thrust {
    void scan(int n, int *odata, const int *idata);
	void radixSort(int n, int *odata, const int *idata);
}
}
