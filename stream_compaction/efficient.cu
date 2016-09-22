#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#define MEASURE_EXEC_TIME
#include "efficient.h"
#include <vector>

#define USING_SHARED_MEMORY

#ifdef USING_SHARED_MEMORY
#define MAX_SEGMENT_SIZE 1024
#define NUM_SEG(x, ss) (((x) + (ss) - 1) / (ss))
#define ROUND_SEG_SIZE(x, ss) (NUM_SEG(x, (ss)) * (ss))

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(x) ((x) >> LOG_NUM_BANKS)
#endif

namespace StreamCompaction {
	namespace Efficient {

		int nearestMultipleOfTwo(int n)
		{
			int result = 1;
			while (result < n) result <<= 1;
			return result;
		}

#ifdef USING_SHARED_MEMORY
		__global__ void kernScan(int segSize, int * __restrict__ blockSums, int * __restrict__ odata, const int * __restrict__ idata)
		{
			extern __shared__ int temp[];

			const int base = blockIdx.x * segSize;
			const int tid = threadIdx.x;
			const int i1 = 2 * tid + 1;
			const int i2 = 2 * tid + 2;
			int offset = 1;
			int ai, bi;

			// cache data
			int gidx1 = base + tid;
			int gidx2 = gidx1 + blockDim.x;
			int lidx1 = tid + CONFLICT_FREE_OFFSET(tid);
			int lidx2 = tid + (segSize >> 1) + CONFLICT_FREE_OFFSET(tid + (segSize >> 1));
			temp[lidx1] = idata[gidx1];
			temp[lidx2] = idata[gidx2];

			// up sweep
			for (int d = segSize >> 1; d > 0; d >>= 1)
			{
				__syncthreads();

				if (tid < d)
				{
					ai = offset * i1 - 1;
					bi = offset * i2 - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);

					temp[bi] += temp[ai];
				}

				offset *= 2;
			}

			if (tid == 0)
			{
				int idx = segSize - 1 + CONFLICT_FREE_OFFSET(segSize - 1);
				if (blockSums) blockSums[blockIdx.x] = temp[idx];
				temp[idx] = 0;
			}

			// down sweep
			for (int d = 1; d < segSize; d *= 2)
			{
				offset >>= 1;
				__syncthreads();

				if (tid < d)
				{
					ai = offset * i1 - 1;
					bi = offset * i2 - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);

					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}

			__syncthreads();

			odata[gidx1] = temp[lidx1];
			odata[gidx2] = temp[lidx2];
		}

		__global__ void kernPerSegmentAdd(int segSize, int * __restrict__ odata, const int * __restrict__ blockSums)
		{
			int bid = blockIdx.x;
			int tid = threadIdx.x;
			int writeIdx1 = bid * segSize + 2 * tid;
			int writeIdx2 = writeIdx1 + 1;

			int sum = blockSums[bid];
			odata[writeIdx1] += sum;
			odata[writeIdx2] += sum;
		}

		inline int computeSegmentSize(int n)
		{
			return n > (MAX_SEGMENT_SIZE >> 1) ? MAX_SEGMENT_SIZE : nearestMultipleOfTwo(n);
		}

#ifdef MEASURE_EXEC_TIME
		float scanHelper(int segSize, int n, int *odata_dev, const int *idata_dev)
#else
		void scanHelper(int segSize, int n, int *odata_dev, const int *idata_dev)
#endif
		{
			// determine segment size
			int threadsPerBlock = segSize >> 1;
			int numBlocks = NUM_SEG(n, segSize);

			int *iblockSums = 0, *oblockSums = 0;
			int segSizeNextLevel;
			if (numBlocks > 1)
			{
				segSizeNextLevel = computeSegmentSize(numBlocks);
				size_t kBlockSumsSize = ROUND_SEG_SIZE(numBlocks, segSizeNextLevel) * sizeof(int);
				cudaMalloc(&iblockSums, kBlockSumsSize);
				cudaMalloc(&oblockSums, kBlockSumsSize);
				cudaMemset(iblockSums, 0, kBlockSumsSize);
			}

#ifdef MEASURE_EXEC_TIME
			float execTime = 0.f;
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
#endif

			kernScan << <numBlocks, threadsPerBlock, (segSize + CONFLICT_FREE_OFFSET(segSize - 1)) * sizeof(int) >> >(segSize, iblockSums, odata_dev, idata_dev);

#ifdef MEASURE_EXEC_TIME
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&execTime, start, stop);
#endif

			if (numBlocks > 1)
			{
#ifdef MEASURE_EXEC_TIME
				execTime += scanHelper(segSizeNextLevel, numBlocks, oblockSums, iblockSums);

				cudaEventRecord(start);
				kernPerSegmentAdd << <numBlocks, threadsPerBlock >> >(segSize, odata_dev, oblockSums);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				float et = 0.f;
				cudaEventElapsedTime(&et, start, stop);
				execTime += et;
#else
				scanHelper(segSizeNextLevel, numBlocks, oblockSums, iblockSums);
				kernPerSegmentAdd << <numBlocks, threadsPerBlock >> >(segSize, odata_dev, oblockSums);
#endif

				cudaFree(iblockSums);
				cudaFree(oblockSums);
			}

#ifdef MEASURE_EXEC_TIME
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			return execTime;
#endif
		}
#else
		__global__ void kernScanUpSweepOneLevel(int offset, int numActiveThreads, int *iodata)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			if (tid >= numActiveThreads)
			{
				return;
			}

			if (numActiveThreads == 1) // last level
			{
				iodata[2 * offset - 1] = 0;
				return;
			}

			int i1 = 2 * tid + 1;
			int i2 = i1 + 1;
			int ai, bi;

			ai = offset * i1 - 1;
			bi = offset * i2 - 1;
			iodata[bi] += iodata[ai];
		}

		__global__ void kernScanDownSweepOneLevel(int offset, int numActiveThreads, int *iodata)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			if (tid >= numActiveThreads)
			{
				return;
			}

			int i1 = 2 * tid + 1;
			int i2 = i1 + 1;
			int ai, bi;

			ai = offset * i1 - 1;
			bi = offset * i2 - 1;
			int t = iodata[ai];
			iodata[ai] = iodata[bi];
			iodata[bi] += t;
		}
#endif

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
#ifdef MEASURE_EXEC_TIME
		float scan(int n, int *odata, const int *idata)
		{
			if (n <= 0 || !odata || !idata || odata == idata)
			{
				return -1;
			}
#else
		void scan(int n, int *odata, const int *idata)
		{
			if (n <= 0 || !odata || !idata || odata == idata)
			{
				return;
			}
#endif
#ifdef USING_SHARED_MEMORY
			int segSize = computeSegmentSize(n);
			const size_t kDevArraySizeInByte = ROUND_SEG_SIZE(n, segSize) * sizeof(int);
			int *odata_dev = 0;
			int *idata_dev = 0;

			cudaMalloc(&odata_dev, kDevArraySizeInByte);
			cudaMalloc(&idata_dev, kDevArraySizeInByte);
			cudaMemset(idata_dev, 0, kDevArraySizeInByte);
			cudaMemcpy(idata_dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);

#ifdef MEASURE_EXEC_TIME
			float execTime = scanHelper(segSize, n, odata_dev, idata_dev);
#else
			scanHelper(segSize, n, odata_dev, idata_dev);
#endif

			cudaMemcpy(odata, odata_dev, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(odata_dev);
			cudaFree(idata_dev);
			cudaDeviceSynchronize();

#ifdef MEASURE_EXEC_TIME
			return execTime;
#endif
#else
			const int paddedSize = nearestMultipleOfTwo(n);
			const size_t kDevArraySizeInByte = paddedSize * sizeof(int);
			int *iodata_dev = 0;

			cudaMalloc(&iodata_dev, kDevArraySizeInByte);
			cudaMemset(iodata_dev, 0, kDevArraySizeInByte);
			cudaMemcpy(iodata_dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);

#ifdef MEASURE_EXEC_TIME
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
#endif

			const int threadsPerBlock = 256;
			const int numLevels = ilog2ceil(n);
			int numActiveThreads = paddedSize >> 1;
			int offset = 1;

			// up sweep
			for (int i = 0; i < numLevels; ++i)
			{
				int numBlocks = (numActiveThreads + threadsPerBlock - 1) / threadsPerBlock;
				kernScanUpSweepOneLevel << <numBlocks, threadsPerBlock >> >(offset, numActiveThreads, iodata_dev);
				numActiveThreads >>= 1;
				offset *= 2;
			}

			// down sweep
			numActiveThreads = 1;
			for (int i = 0; i < numLevels; ++i)
			{
				offset >>= 1;
				int numBlocks = (numActiveThreads + threadsPerBlock - 1) / threadsPerBlock;
				kernScanDownSweepOneLevel << <numBlocks, threadsPerBlock >> >(offset, numActiveThreads, iodata_dev);
				numActiveThreads <<= 1;
			}

#ifdef MEASURE_EXEC_TIME
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float millisceconds = 0;
			cudaEventElapsedTime(&millisceconds, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
#endif

			cudaMemcpy(odata, iodata_dev, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(iodata_dev);
			cudaDeviceSynchronize();

#ifdef MEASURE_EXEC_TIME
			return millisceconds;
#endif
#endif
		}

		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param odata  The array into which to store elements.
		 * @param idata  The array of elements to compact.
		 * @returns      The number of elements remaining after compaction.
		 */
		int compact(int n, int *odata, const int *idata)
		{
			if (n <= 0 || !odata || !idata || odata == idata)
			{
				return -1;
			}

			using StreamCompaction::Common::kernMapToBoolean;
			using StreamCompaction::Common::kernScatter;

			int *idata_dev = 0;
			int *odata_dev = 0;
			int *bools_dev = 0;
			int *indices_dev = 0;

			int segSize = computeSegmentSize(n);
			const size_t kBoolsSizeInByte = ROUND_SEG_SIZE(n, segSize) * sizeof(int);
			const size_t kIndicesSizeInByte = kBoolsSizeInByte;

			cudaMalloc(&idata_dev, n * sizeof(int));
			cudaMalloc(&bools_dev, kBoolsSizeInByte);
			cudaMalloc(&indices_dev, kIndicesSizeInByte);

			cudaMemcpy(idata_dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(bools_dev, 0, kBoolsSizeInByte);

			const int threadsPerBlock = 256;
			int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
			kernMapToBoolean << <numBlocks, threadsPerBlock >> >(n, bools_dev, idata_dev);

			scanHelper(segSize, n, indices_dev, bools_dev);

			int numElemRemained;
			cudaMemcpy(&numElemRemained, indices_dev + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
			numElemRemained += idata[n - 1] ? 1 : 0;
			cudaMalloc(&odata_dev, numElemRemained * sizeof(int));

			kernScatter<<<numBlocks, threadsPerBlock>>>(n, odata_dev, idata_dev, bools_dev, indices_dev);

			cudaMemcpy(odata, odata_dev, numElemRemained * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(idata_dev);
			cudaFree(odata_dev);
			cudaFree(bools_dev);
			cudaFree(indices_dev);

			return numElemRemained;
		}

	}
}
