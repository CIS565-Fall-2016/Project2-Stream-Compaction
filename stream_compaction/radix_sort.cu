#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <exception>
#include <string>

#define MEASURE_EXEC_TIME
#include <stream_compaction/efficient.h>
#include "radix_sort.h"


namespace std
{
	class InvalidArgument : public exception
	{
	public:
		virtual const char *what() const throw()
		{
			return "One or more invalid arguments detected";
		}
	};
}


namespace ParallelRadixSort
{
	template <class T>
	__global__ void kernClassify(uint32_t n, T mask,
		uint32_t * __restrict__ notbools, uint32_t * __restrict__ bools, const T * __restrict__ idata)
	{
		uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid >= n) return;

		uint32_t t = static_cast<uint32_t>((idata[tid] & mask) == 0);
		notbools[tid] = t;
		bools[tid] = t ^ 0x1;
	}

	template <class T>
	__global__ void kernScatter(uint32_t n,
		uint32_t * __restrict__ nobools, uint32_t * __restrict__ noindices, uint32_t * __restrict__ yesindices,
		T * __restrict__ odata, const T * __restrict__ idata)
	{
		uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid >= n) return;

		uint32_t isBitZero = nobools[tid];
		uint32_t idx;

		if (isBitZero) idx = noindices[tid]; else idx = yesindices[tid];

		odata[idx] = idata[tid];
	}

#ifdef MEASURE_EXEC_TIME
	template <class T>
	float sort(int n, T *odata, const T *idata, T bitMask, bool lsb)
#else
	template <class T>
	void sort(int n, T *odata, const T *idata, T bitMask, bool lsb)
#endif
	{
		if (n <= 0 || !odata || !idata)
		{
			throw std::InvalidArgument();
		}

		int segSize = StreamCompaction::Efficient4::computeSegmentSize(2 * n);
		const size_t kDevArraySizeInByte = StreamCompaction::Efficient4::computeActualMemSize(2 * n);

		T *idata_dev = 0;
		T *odata_dev = 0;
		uint32_t *noyes_bools_dev = 0;
		uint32_t *indices_dev = 0;

		cudaMalloc(&idata_dev, n * sizeof(T));
		cudaMalloc(&odata_dev, n * sizeof(T));
		cudaMalloc(&noyes_bools_dev, kDevArraySizeInByte);
		cudaMalloc(&indices_dev, kDevArraySizeInByte);

		cudaMemcpy(idata_dev, idata, n * sizeof(T), cudaMemcpyHostToDevice);

		const int threadsPerBlock = 256;
		int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
		int numBits = 8 * sizeof(T);
		T mask = lsb ? 1 : (1 << (numBits - 1));

#ifdef MEASURE_EXEC_TIME
		float execTime = 0.f;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		for (int i = 0; i < numBits; ++i)
		{
			if (!(bitMask & mask)) continue; // do not consider this bit

			kernClassify<<<numBlocks, threadsPerBlock>>>(n, mask, noyes_bools_dev, noyes_bools_dev + n, idata_dev);

			StreamCompaction::Efficient4::scanHelper(2 * n, indices_dev, noyes_bools_dev);

			kernScatter<<<numBlocks, threadsPerBlock>>>(n, noyes_bools_dev, indices_dev, indices_dev + n, odata_dev, idata_dev);

			if (lsb) mask <<= 1; else mask >>= 1;

			T *tmp = odata_dev;
			odata_dev = idata_dev;
			idata_dev = tmp;
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&execTime, start, stop);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
#else
		for (int i = 0; i < numBits; ++i)
		{
			if (!(bitMask & mask)) continue; // do not consider this bit

			kernClassify << <numBlocks, threadsPerBlock >> >(n, mask, noyes_bools_dev, noyes_bools_dev + n, idata_dev);

			StreamCompaction::Efficient::scanHelper(segSize, 2 * n, indices_dev, noyes_bools_dev);

			kernScatter << <numBlocks, threadsPerBlock >> >(n, noyes_bools_dev, indices_dev, indices_dev + n, odata_dev, idata_dev);

			if (lsb) mask <<= 1; else mask >>= 1;

			T *tmp = odata_dev;
			odata_dev = idata_dev;
			idata_dev = tmp;
		}
#endif

		cudaMemcpy(odata, idata_dev, n * sizeof(T), cudaMemcpyDeviceToHost);
		cudaFree(idata_dev);
		cudaFree(odata_dev);
		cudaFree(noyes_bools_dev);
		cudaFree(indices_dev);

#ifdef MEASURE_EXEC_TIME
		return execTime;
#endif
	}

	// Since template definition is not visible to users (main.obj in this case),
	// we need to explicitly tell the compiler to generate all the template implementations
	// that will be used later
#ifdef MEASURE_EXEC_TIME
	template float sort<uint32_t>(int n, uint32_t *odata, const uint32_t *idata, uint32_t bitMask, bool lsb);
#else
	template void sort<uint32_t>(int n, uint32_t *odata, const uint32_t *idata, uint32_t bitMask, bool lsb);
#endif

#ifdef MEASURE_EXEC_TIME
	template <class T>
	float thrustSort(int n, T *odata, const T *idata)
	{
		if (n <= 0 || !odata || !idata)
		{
			throw std::InvalidArgument();
		}

		T *iodata_dev = 0;

		cudaMalloc(&iodata_dev, n * sizeof(T));
		cudaMemcpy(iodata_dev, idata, n * sizeof(T), cudaMemcpyHostToDevice);

		thrust::device_ptr<T> thrust_iodata(iodata_dev);

		float execTime;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		thrust::stable_sort(thrust_iodata, thrust_iodata + n);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&execTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaMemcpy(odata, iodata_dev, n * sizeof(T), cudaMemcpyDeviceToHost);
		cudaFree(iodata_dev);

		return execTime;
	}

	template float thrustSort(int n, uint32_t *odata, const uint32_t *idata);
#endif
}