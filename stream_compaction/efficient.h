#pragma once

namespace StreamCompaction {
namespace Efficient {
	void scan(int n, int *odata, const int *idata);
	void scan_dev(int n, int *dev_data);

    int compact(int n, int *odata, const int *idata);
}
}
