CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ruoyu Fan
* Tested on: Windows 10 x64, i7-6700K @ 4.00GHz 16GB, GTX 970 4096MB (girlfriend's machine)
  * compiled with Visual Studio 2013 and CUDA 7.5

NOTE: if the program crashes when entering the test for naive sort, try reducing the array sizes in `main.cpp`. (`SIZE` and `SORT_SIZE`.)

![preview](/screenshots/preview_optimized.gif)

### Description

#### Things I have done

* Implemented __CPU scan and compaction__, __compaction__, __GPU naive scan__, __GPU work-efficient scan__, __GPU work-efficient compaction__, __GPU radix sort (extra)__, and compared my scan algorithms with thrust implemention

* I optimized my work efficient scan, and __speed is increased to 270%__ of my original implementation.

* I also wrote an __invlusive version__ of __work-efficient scan__ - because i misunderstood the requirement at first! The difference of the inclusive method is that it creates a buffer that is 1 element larger and swap the last(0) and and second last elements before downsweeping. Although I corrected my implemention to exclusive scan, the inclusive scan can still be called by passing ScanType::inclusive to scan_implenmention method in efficient.cu.

* __Radix sort__ assumes inputs are between [0, a_given_maximum) . I compared my radix sort with std::sort and thrust's unstable and stable sort.

* I added a helper class `PerformanceTimer` in common.h which is used to do performance measurement.


#### Original Questions
```
* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * (You shouldn't compare unoptimized implementations to each other!)

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * You should use CUDA events for timing GPU code. Be sure **not** to include
    any *initial/final* memory operations (`cudaMalloc`, `cudaMemcpy`) in your
    performance measurements, for comparability. Note that CUDA events cannot
    time CPU code.
  * You can use the C++11 `std::chrono` API for timing CPU code. See this
    [Stack Overflow answer](http://stackoverflow.com/a/23000049) for an example.
    Note that `std::chrono` may not provide high-precision timing. If it does
    not, you can either use it to time many iterations, or use another method.
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.
```

* Please refer to __Performance__ section.

```
* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?
```

* I notice that I couldn't get a good measurement for scan and sort of __Thrust__. I have trouble measuring `thrust::exclusive` with std::chrono, while I can use `std::chrono` to measure `thrust::scan` but the results from CUDA events seems off.

* With max possible value increased (so does MSB), besides the run time of radix sort, that of std::sort also increased significantly, which is unexpected. Does bit length also affect the time for read/write or other operations? or the implementation of std::sort<int> use bit level information?

* I think the bottleneck for blocksize is the warp size inside GPU.

* My original work-efficient scan implementation was slower than CPU scan, that was because I wasted too many of my threads, please refer to __Optimization__ section below.

```
* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.
```

* The tests are done with arrays of lengths `2^26` (67108864) and `2^26-3` (67108861). The array generation uses current time as random seed.

* I added tests for __radix sort__, which compares with `std::sort` as well as __Thrust__'s stable and unstable sorts


#### Performance

##### Blocksize

When block size is smaller than 16, my application suffers from performance drop, which is recorded in `test_results` folder. I decided to just use `cudaOccupancyMaxPotentialBlockSize` for each device functions, which is almost 1024 on my computer.

| Block size | Naïve Scan(ms) | Work-efficient Scan (ms) | Reference CPU Scan (ms) |
|------------|----------------|--------------------------|-------------------------|
| 4          | 755.433        | 76.5717                  | 134.408                 |
| 8          | 379.897        | 52.1942                  | 134.408                 |
| 16         | 212.542        | 44.3224                  | 134.408                 |
| 32         | 133.116        | 43.4925                  | 134.408                 |
| 64         | 114.7          | 44.7172                  | 134.408                 |
| 128        | 113.802        | 44.7841                  | 134.408                 |
| 256        | 114.575        | 45.3902                  | 134.408                 |
| 512        | 114.593        | 44.3084                  | 134.408                 |
| 1024       | 113.867        | 44.2941                  | 134.408                 |

##### Array length

All the test are done with block size of 1024. The possible max value for sorting is 100.

__Scan__ : this work-efficient scan implementation is faster than cpu scan on large input but slower than Thrust's

| Input array length power of 2 | CPU Scan (ms) | Naïve Scan (ms) | Work-efficient scan (ms) | Thrust scan (ms) (CUDA event) |
|-------------------------------|---------------|-----------------|--------------------------|-------------------------------|
| 12                            | 0             | 0.047872        | 0.110784                 | 0.028192                      |
| 16                            | 0             | 0.09216         | 0.180064                 | 0.166432                      |
| 20                            | 0             | 1.29402         | 0.818528                 | 0.30768                       |
| 24                            | 23.5625       | 25.7094         | 11.3379                  | 2.00646                       |
| 26                            | 159.925       | 133.61          | 44.4511                  | 7.7283                        |

__Compaction__ : this work-efficient compaction implementation is faster than cpu's

| Input array length power of 2 | CPU compact (ms) | CPU scan compact  (ms) | Work-efficient compact (ms) |
|-------------------------------|------------------|------------------------|-----------------------------|
| 12                            | 0                | 0                      | 0.118368                    |
| 16                            | 0                | 0                      | 0.193408                    |
| 20                            | 0                | 0                      | 0.966944                    |
| 24                            | 39.0743          | 38.6241                | 13.3034                     |
| 26                            | 155.406          | 422.524                | 53.2659                     |

__Sort__ : radix sort on GPU is faster than std::sort for `int`s while roughly the same as Thrust's implementations

| Input array length power of 2 | std::sort (ms) | Radix sort (ms) | Thrust unstable sort (std::chrono) | Thrust stable sort (std::chrono) |
|-------------------------------|----------------|-----------------|------------------------------------|----------------------------------|
| 12                            | 0              | 0.883328        | 0                                  | 0                                |
| 16                            | 0              | 1.44288         | 0                                  | 0                                |
| 20                            | 22.1649        | 7.61686         | 0                                  | 0                                |
| 24                            | 378.025        | 105.222         | 109.329                            | 100.279                          |
| 26                            | 1481.8         | 419.345         | 434.399                            | 423.933                          |

##### Data maximum value and radix sort

All the test are done with block size of 1024; array length is 67108864.

| max value | std::sort (ms) | Radix sort (ms) | Thrust unstable sort (std::chrono) | Thrust stable sort (std::chrono) |
|-----------|----------------|-----------------|------------------------------------|----------------------------------|
| 100       | 1481.8         | 419.345         | 434.399                            | 423.933                          |
| 10000     | 3564.1         | 1023.97         | 580.816                            | 587.524                          |

With max possible value increased, besides the run time of radix sort, that of std::sort and thrust sorts also increased, unexpectedly.


#### Sample Output

```
CIS-565 HW2 CUDA Stream Compaction Test (Ruoyu Fan)
    Block size for naive scan: 1024
    Block size for up-sweep: 1024
    Block size for down-sweep: 1024
    Block size for boolean mapping: 1024
    Block size for scattering: 1024
    Block sizes for radix sort: 1024 1024 1024 1024

****************
** SCAN TESTS **
****************
Array size (power of two): 67108864
Array size (non-power of two): 67108861
    [   8  18  37  41  15  25  27   8  36  28  13  40  24 ...  35   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 134.408ms    (std::chrono Measured)
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625502 1643625537 ]
==== cpu scan, non-power-of-two ====
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625408 1643625440 ]
   elapsed time: 149.901ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625502 1643625537 ]
   elapsed time: 113.867ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625408 1643625440 ]
   elapsed time: 113.687ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625502 1643625537 ]
   elapsed time: 44.2491ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625408 1643625440 ]
   elapsed time: 44.3104ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625502 1643625537 ]
   elapsed time: 7.73741ms    (CUDA Measured)
   elapsed time: 0ms    (std::chrono Measured)
    passed
==== thrust scan, non-power-of-two ====
    [   0   8  26  63 104 119 144 171 179 215 243 256 296 ... 1643625408 1643625440 ]
   elapsed time: 7.74371ms    (CUDA Measured)
   elapsed time: 0ms    (std::chrono Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
Array size (power of two): 67108864
Array size (non-power of two): 67108861
    [   0   1   0   3   3   1   2   1   1   2   1   0   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
    [   1   3   3   1   2   1   1   2   1   3   3   1   3 ...   2   3 ]
   elapsed time: 155.403ms    (std::chrono Measured)
    passed
==== cpu compact without scan, non-power-of-two ====
    [   1   3   3   1   2   1   1   2   1   3   3   1   3 ...   2   2 ]
   elapsed time: 154.901ms    (std::chrono Measured)
    passed
==== cpu compact with scan ====
    [   1   3   3   1   2   1   1   2   1   3   3   1   3 ...   2   3 ]
   elapsed time: 421.621ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
    [   1   3   3   1   2   1   1   2   1   3   3   1   3 ...   2   3 ]
   elapsed time: 54.2043ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
    [   1   3   3   1   2   1   1   2   1   3   3   1   3 ...   2   2 ]
   elapsed time: 54.1137ms    (CUDA Measured)
    passed

*****************************
** RADIX SORT TESTS **
*****************************
Array size (power of two): 67108864
Array size (non-power of two): 67108861
Max value: 100
    [  78  22  68  49  66  85  83  63  52  58  25   5  35 ...  84   0 ]
==== std::sort, power-of-two ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
   elapsed time: 1522.95ms    (std::chrono Measured)
==== thrust unstable sort, power-of-two ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
   elapsed time: 429.389ms    (std::chrono Measured)
   elapsed time: 0.001184ms    (CUDA Measured)
    passed
==== thrust stable sort, power-of-two ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
   elapsed time: 418.915ms    (std::chrono Measured)
   elapsed time: 0.001216ms    (CUDA Measured)
    passed
==== radix sort, power-of-two ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
   elapsed time: 419.691ms    (CUDA Measured)
    passed
==== std::sort, non power-of-two ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
   elapsed time: 1516.39ms    (std::chrono Measured)
==== radix sort, non power-of-two ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
   elapsed time: 416.676ms    (CUDA Measured)
    passed
```


#### Optimization

##### Run less threads for work-efficient scan

For work-efficient scan, my original implementation was using the same of amount of threads for every up sweep and down sweeps. Then I optimized it by using only necessary amount of threads for each iteration.

The performance for scanning an array of length 67108861 using work-efficient approach boosted __from ~120.5ms to ~44.4ms__, which is __270% speed__ of my original approach. You can see the data in the files under __test_results/__ folder

Original implementation:

```c++
// running unnecessary threads
__global__ void kernScanUpSweepPass(int N, int add_distance, int* buffer)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    if ((index + 1) % (add_distance * 2) == 0)
    {
        buffer[index] = buffer[index] + buffer[index - add_distance];
    }
}

__global__ void kernScanDownSweepPass(int N, int distance, int* buffer)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) { return; }

    if ((index + 1) % (distance * 2) == 0)
    {
        auto temp = buffer[index - distance];
        buffer[index - distance] = buffer[index];
        buffer[index] = temp + buffer[index];
    }
}
```

New implementation:

```c++
// optimized: only launch necessary amount of threads in host code
__global__ void kernScanUpSweepPass(int max_thread_index, int add_distance, int* buffer)
{
    auto tindex = threadIdx.x + blockIdx.x * blockDim.x;

    if (tindex >= max_thread_index) { return; }

    // I encountered overflow problem with index < N here so I changed to tindex < max_thread_index
    size_t index = (add_distance * 2) * (1 + tindex) - 1;

    buffer[index] = buffer[index] + buffer[index - add_distance];
}

// optimized: only launch necessary amount of threads in host code
__global__ void kernScanDownSweepPass(int max_thread_index, int distance, int* buffer)
{
    auto tindex = threadIdx.x + blockIdx.x * blockDim.x;

    if (tindex >= max_thread_index) { return; }

    size_t index = (distance * 2) * (1 + tindex) - 1;  

    auto temp = buffer[index - distance];
    buffer[index - distance] = buffer[index];
    buffer[index] = temp + buffer[index];
}
```

And I calculated the number of threads needed as well as the maximum thread index for every up-sweep and down-sweep pass.

Originally I was still using length of buffer as first parameter, but when I was calculating indices for a thread by using the condition of `(distance * 2) * (1 + tindex) - 1 > N`. There can come some weird result because of the multiplication result is out of bound (even for `size_t` - it took me 2 hours to debug that). So lessons learned, and I'll use more `n > b/a` instead of `a*n > b` as condition in the future.

##### Helper class for performance measurement

I create a RAII `PerformanceTimer` class for performance measurement. Which is like:

```c++
/**
* This class is used for timing the performance
* Uncopyable and unmovable
*/
class PerformanceTimer
{
public:
    PerformanceTimer()
    {
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_end);
    }

    ~PerformanceTimer()
    {
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_end);
    }

    void startCpuTimer()
    {
        if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
        cpu_timer_started = true;

        time_start_cpu = std::chrono::high_resolution_clock::now();
    }

    void endCpuTimer()
    {
        time_end_cpu = std::chrono::high_resolution_clock::now();

        if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

        std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
        prev_elapsed_time_cpu_milliseconds =
            static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

        cpu_timer_started = false;
    }

    void startGpuTimer()
    {
        if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
        gpu_timer_started = true;

        cudaEventRecord(event_start);
    }

    void endGpuTimer()
    {
        cudaEventRecord(event_end);
        cudaEventSynchronize(event_end);

        if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

        cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
        gpu_timer_started = false;
    }

    float getCpuElapsedTimeForPreviousOperation() noexcept
    {
        return prev_elapsed_time_cpu_milliseconds;
    }

    float getGpuElapsedTimeForPreviousOperation() noexcept
    {
        return prev_elapsed_time_gpu_milliseconds;
    }

    // remove copy and move functions
    PerformanceTimer(const PerformanceTimer&) = delete;
    PerformanceTimer(PerformanceTimer&&) = delete;
    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
    PerformanceTimer& operator=(PerformanceTimer&&) = delete;

private:
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;

    using time_point_t = std::chrono::high_resolution_clock::time_point;
    time_point_t time_start_cpu;
    time_point_t time_end_cpu;

    bool cpu_timer_started = false;
    bool gpu_timer_started = false;

    float prev_elapsed_time_cpu_milliseconds = 0.f;
    float prev_elapsed_time_gpu_milliseconds = 0.f;
};
```

And inside a module I have:

```c++
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer& timer()
{
    // not thread-safe
    static PerformanceTimer timer;
    return timer;
}
```

Therefore, I can use

```c++
void someFunc()
{
    allocateYourBuffers()

    timer().startGpuTimer();

    doYourGpuScan();

    timer().endGpuTimer();

    endYourJob();
}
```

and

```c++
timer().getGpuElapsedTimeForPreviousOperation();
```

to get the measured elapsed time for the operation.
