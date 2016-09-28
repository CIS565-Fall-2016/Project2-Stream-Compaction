#pragma once

namespace StreamCompaction {
namespace CPU {
    double scan(int n, int *odata, const int *idata);

    int compactWithoutScan(int n, int *odata, const int *idata, double &time);

    int compactWithScan(int n, int *odata, const int *idata, double &time);
}
}
