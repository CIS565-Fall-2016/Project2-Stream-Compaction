#pragma once

namespace StreamCompaction {
namespace Thrust {
    void scan(int n, int *odata, const int *idata);
	void sort(int n, int *odata, const int *idata);
}
}
