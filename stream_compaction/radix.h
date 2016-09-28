#pragma once


namespace StreamCompaction {
namespace Radix {
    enum gputesttype { CPU, NAIVE, THRUST };

    void radixCPU(int n, int *odata, const int *idata);
    void radixGPU(int n, int *odata, const int *idata, gputesttype testtype);
}
}
