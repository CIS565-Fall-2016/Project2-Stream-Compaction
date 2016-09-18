#pragma once

#include "common.h"

namespace StreamCompaction {
namespace Thrust {
    StreamCompaction::Common::PerformanceTimer& timer();
    void scan(int n, int *odata, const int *idata);
    void unstableSort(int* start, int* end);
    void stableSort(int* start, int* end);
}
}
