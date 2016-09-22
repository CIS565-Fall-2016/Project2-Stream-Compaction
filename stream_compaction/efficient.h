#pragma once

namespace StreamCompaction {
namespace Efficient {
#ifdef MEASURE_EXEC_TIME
    float scan(int n, int *odata, const int *idata);
#else
	void scan(int n, int *odata, const int *idata);
#endif

    int compact(int n, int *odata, const int *idata);
}
}
