#pragma once

namespace StreamCompaction {
namespace Naive {
    extern double last_runtime;
    extern int blkSize;

    void scan(int n, int *odata, const int *idata);
}
}
