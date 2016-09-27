#pragma once

namespace StreamCompaction {
namespace CPU {
    void scan(int n, int *odata, const int *idata);

    int compactWithoutScan(int n, int *odata, const int *idata);

	int scatter(int n, int *odata, const int *idata, const int *idataChanged, const int *exclusivePreSum);
	
    int compactWithScan(int n, int *odata, const int *idata);
}
}
