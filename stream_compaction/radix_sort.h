#pragma once

#include "common.h"

namespace StreamCompaction {
namespace RadixSort {
    StreamCompaction::Common::PerformanceTimer& timer();
    void radixSort(int* start, int* end, int max_value);
}
}
