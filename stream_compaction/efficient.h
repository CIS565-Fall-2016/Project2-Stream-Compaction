#pragma once

namespace StreamCompaction {
namespace Efficient {
    extern double last_runtime;
    extern int blkSize;

    void scan(int n, int *odata, const int *idata);

    int compact(int n, int *odata, const int *idata);
}
}
