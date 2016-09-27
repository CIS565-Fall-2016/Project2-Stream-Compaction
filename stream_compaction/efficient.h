#pragma once

#define USING_SHARED_MEMORY

#ifdef USING_SHARED_MEMORY
#define MAX_SEGMENT_SIZE 1024
#define NUM_SEG(x, ss) (((x) + (ss) - 1) / (ss))
#define ROUND_SEG_SIZE(x, ss) (NUM_SEG(x, (ss)) * (ss))

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(x) ((x) >> LOG_NUM_BANKS)
#endif

namespace StreamCompaction
{
	namespace Efficient
	{
#ifdef MEASURE_EXEC_TIME
		float scan(int n, int *odata, const int *idata);

		int compact(int n, int *odata, const int *idata, float *pExecTime);
#else
		void scan(int n, int *odata, const int *idata);

		int compact(int n, int *odata, const int *idata);
#endif

		void scanHelper(int segSize, int n, int *odata_dev, const int *idata_dev);

		inline int nearestMultipleOfTwo(int n)
		{
			int result = 1;
			while (result < n) result <<= 1;
			return result;
		}

		inline int computeSegmentSize(int n)
		{
			return n > (MAX_SEGMENT_SIZE >> 1) ? MAX_SEGMENT_SIZE : nearestMultipleOfTwo(n);
		}

		inline size_t alignedSize(size_t sizeInBytes, size_t alignmentInBytes)
		{
			return (sizeInBytes + alignmentInBytes - 1) / alignmentInBytes * alignmentInBytes;
		}

		/**
		 * When data size is large, multiple passes are necessary to compute the
		 * final result and extra device memory is required to store intermediate
		 * results.
		 *
		 * Use the result from this function to cudaMalloc the input and output
		 * buffers passed to scanHelper which assumes that I/O device buffers have
		 * sufficient size.
		 *
		 * Assuming address start at a 256-byte boundary
		 */
		template<class T>
		inline size_t computeActualMemSize(int n)
		{
			const size_t kMemAlignmentInBytes = 256; // the alignment CUDA driver used

			int segSize = computeSegmentSize(n);
			int numSegs = NUM_SEG(n, segSize);
			size_t total = alignedSize(numSegs * segSize * sizeof(T), kMemAlignmentInBytes);

			while (numSegs > 1)
			{
				segSize = computeSegmentSize(numSegs);
				numSegs = NUM_SEG(numSegs, segSize);
				size_t extra = alignedSize(numSegs * segSize * sizeof(T), kMemAlignmentInBytes);
				total += extra;
			}

			return total;
		}
	}

	namespace Efficient4
	{
		float scan(int n, unsigned *odata, const unsigned *idata);
	}
}
