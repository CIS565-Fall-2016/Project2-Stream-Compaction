#pragma once

namespace StreamCompaction {
namespace Thrust {
    extern double last_runtime;

    void scan(int n, int *odata, const int *idata);
}
}
