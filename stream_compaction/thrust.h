#pragma once

namespace StreamCompaction {
namespace Thrust {
    void scan(int n, int *odata, const int *idata);
    float timeThrust(int n, int *odata, const int *idata);
}
}
