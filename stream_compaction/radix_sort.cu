#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <exception>
#include <string>

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

	template <class T>
	void sort(int n, T *odata, const T *idata, T bitMask, bool lsb)
	{
		if (n <= 0 || !odata || !idata)
		{
			throw std::InvalidArgument();
		}

		T *idata_dev = 0;
		T *odata_dev = 0;
		uint32_t *noyes_bools_dev = 0;
		uint32_t *indices_dev = 0;

		cudaMalloc(&idata_dev, n * sizeof(T));
		cudaMalloc(&odata_dev, n * sizeof(T));
		cudaMalloc(&noyes_bools_dev, 2 * n * sizeof(uint32_t));
		cudaMalloc(&indices_dev, 2 * n * sizeof(uint32_t));

		cudaMemcpy(idata_dev, idata, n * sizeof(T), cudaMemcpyHostToDevice);

		thrust::device_ptr<uint32_t> thrust_bools(noyes_bools_dev);
		thrust::device_ptr<uint32_t> thrust_indices(indices_dev);

		const int threadsPerBlock = 256;
		int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
		int numBits = 8 * sizeof(T);
		T mask = lsb ? 1 : (1 << (numBits - 1));

		for (int i = 0; i < numBits; ++i)
		{
			if (!(bitMask & mask)) continue; // do not consider this bit

			kernClassify << <numBlocks, threadsPerBlock >> >(n, mask, noyes_bools_dev, noyes_bools_dev + n, idata_dev);

			thrust::exclusive_scan(thrust_bools, thrust_bools + 2 * n, thrust_indices);

			kernScatter << <numBlocks, threadsPerBlock >> >(n, noyes_bools_dev, indices_dev, indices_dev + n, odata_dev, idata_dev);

			if (lsb) mask <<= 1; else mask >>= 1;

			T *tmp = odata_dev;
			odata_dev = idata_dev;
			idata_dev = tmp;
		}

		cudaMemcpy(odata, idata_dev, n * sizeof(T), cudaMemcpyDeviceToHost);
		cudaFree(idata_dev);
		cudaFree(odata_dev);
		cudaFree(noyes_bools_dev);
		cudaFree(indices_dev);
	}

	// Since template definition is not visible to users (main.obj in this case),
	// we need to explicitly tell the compiler to generate all the template implementations
	// that will be used later
	template void sort<uint32_t>(int n, uint32_t *odata, const uint32_t *idata, uint32_t bitMask, bool lsb);
}