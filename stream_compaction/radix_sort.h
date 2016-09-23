#pragma once

#include <stdint.h>


namespace ParallelRadixSort
{
	// Only work on unsigned integral types
#ifdef MEASURE_EXEC_TIME
	template <class T>
	float sort(int n, T *odata, const T *idata, T bitMask, bool lsb = true);

	template <class T>
	float thrustSort(int n, T *odata, const T *idata);
#else
	template <class T>
	void sort(int n, T *odata, const T *idata, T bitMask, bool lsb = true);
#endif
}