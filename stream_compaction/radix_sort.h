#pragma once

#include <stdint.h>


namespace ParallelRadixSort
{
	// Only work on unsigned integral types
	template <class T>
	void sort(int n, T *odata, const T *idata, T bitMask, bool lsb = true);
}