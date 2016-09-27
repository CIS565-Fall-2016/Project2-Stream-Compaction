####University of Pennsylvania
####CIS 565: GPU Programming and Architecture

##Project 2 - CUDA Stream Compaction

* Xueyin Wan
* Tested on: Windows 10 x64, i7-6700K @ 4.00GHz 16GB, GTX 970 4096MB (Personal Desktop)
* Compiled with Visual Studio 2013 and CUDA 7.5

**SCREENSHOT**
-------------
**BlockSize : 128**  
**SIZE :  1 << 24**  
**SIZE :  1 << 24**  

![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/XueyinResultOriginal_pow(2%2C24).gif "Performance One") 

###   
**FEATURES I IMPLEMENT**
-------------
```javascript
####Part 1: CPU Scan & Stream Compaction
####Part 2: Naive GPU Scan Algorithm
####Part 3: Work-Efficient GPU Scan & Stream Compaction
####Part 4: Thrust Exclusive Scan using Thrust library
####Part 5: Radix Sort (Extra Credit)
####Part 6: Using std::chrono and CUDA events for comparing the speed of different algorithms 
```
###        

**Dive Into Block Size**
-------------
In order to find the relationship between block size and performance, I modified block size to see different algorithm run time in order to get the optimized block size.   
Below is my chart based on my code:

**Case 1:** 
#####Power of Two number, `SIZE` = 1 << 24 = 16777216
#####All the time recorded are in `ms`.

Block Size | Naïve Scan | Efficient Scan | Thrust Scan| Radix Sort|CPU Scan
---|---|---|---|---|---
16 | 52.045727 | 91.401695 |2.128768|2466.394043|24.0632
32 | 30.109312  | 53.902912 |2.09424|1302.962769|24.0563
64 | 25.546721  | 29.845119 |2.081152|786.153015|24.0908
128 | 25.994272 | 27.808865 |2.255712|741.838257|24.0321
256 | 25.615328 | 27.646433 |2.404192|769.176636|24.064
512 | 25.576256 | 29.840576 |2.256288|791.23053|24.5889
1024 | 25.609535 | 33.565887 |2.211232|860.626038|24.0653



**Case 2:**
#####Non Power of Two number, `SIZE(NPOT)`= 1 << 24 - 3 = 16777213  
#####All the time recorded are in `ms`.
Block Size        | Naïve Scan   | Efficient Scan | Thrust Scan| Radix Sort|CPU Scan|
---|---|---|---|---|---
16 | 45.901855 | 89.639648 |2.094752|2448.440186|42.9234
32 | 30.138912  | 51.030048 |2.29776|1298.191284|43.1142
64 | 25.93968  | 27.795744 |2.011712|788.341675|42.6413
128 | 25.812672 | 24.770847 |2.052608|743.634277|42.6398
256 | 25.627424 | 27.607807 |2.223552|771.496643|41.6099
512 | 25.609535 | 29.848961 |2.146816|803.836487|42.1115
1024 | 26.082048 | 33.715874 |2.04576|851.824646|42.6002


Now let me draw a graph to explicitly show my result :)  
`Notice: ` 
This graph is based on Case 1 result, `Array Size` is Power of Two number, `SIZE` = 1 << 24 = 16777216   
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/ReadMeAboutBlockSizeChoose1.PNG "Chart1")  
###   
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/ReadMeAboutBlockSizeChoose2.PNG "Chart2")

From case 1 and case 2, we could see that when block size is less than 128, the algorithm performance is definitely worse than block size = 128. And after we set block size to 128, we could see that radix sort performance reaches to the highest level. After block size continues to grow, we could notice that Naive Scan, Efficient Scan and Radix Sort are all becoming slower.
So I choose my block size to be `128` in my code.  

**Dive Into Array Size**
-------------
I set block size = `128` in my code, and start to use array size as a parameter to change, in order to compare the performance between different GPU algorithms and CPU algorithm.
Below is my chart based on my code:
 
#####`Blocksize` = 128
#####All the time recorded are in `ms`.
#####Max Value for scan in the array is 50
Array Size | Naïve Scan | Efficient Scan | Thrust Scan|CPU Scan
---|---|---|---|---
2^8 | 0.031904 | 0.11024 |0.021248|0
2^12 | 0.047008  | 0.141728 |0.027616|0 
2^16 | 0.13168  | 0.347968 |0.245728|0.5013 
2^20 | 1.297824 | 1.681472 |0.468608|1.5041 
2^24 | 25.53968 | 27.6632 |2.403232|25.0931 

#####`Blocksize` = 128
#####All the time recorded are in `ms`.
#####Max Value for sort in the array is 2^15
Array Size | Radix Sort | Std::sort  
---|---|---
2^8 | 1.105344 | 0
2^12 | 2.223136  | 0
2^16 | 7.358048  | 4.0105 
2^20 | 42.627296 | 58.1868
2^24 | 749.649841 | 894.4247

Graph for summary:
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/ReadMeAboutArraySizeChoose0.PNG "Chart1")  
###   
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/ReadMeAboutArraySizeChoose1.PNG "Chart2")

From the test result, we could see GPU implementation is slower than CPU's. But when the array size grows, they two become close.    
Thrust Scan is very fast for large array size.    

####     
But for Radix sort, I compare it with std::sort. We could see that when arraysize is small, std::sort is faster.    
However, as array size grows, our radix sort on GPU is much faster than std::sort!  

###Answer to Questions

#### 1. Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.
Done! See above `Dive Into Block Size` part.


#### 2. Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).
Done! See above `Dive Into Array Size` part.     
I use CUDA events for timing GPU code. 
I use std::chrono for timing CPU code.
 
#### 3.To guess at what might be happening inside the Thrust implementation (e.g. allocation, memory copy)
Answer:   
I guess the inner mechnism of THRUST requires some initialization operations.    
After this step, it reaches to better performance.    
(For int numbers it is also implemented in radix sort algorithm.)

#### 4.Can you find the performance bottlenecks? 
Answer:    
Yes. First I want to mention, When we are doing iterations in scan function, and inside each loop is kernal function like Upsweep, downsweep and scan in device. We could see that as the iteration goes on, one phenomena appears:     
There are several threads idling. Since they need to wait those threads which are working to finish there mission, they have to be idling, which causes extra resource allocate.  
My code here to address this problem:
```java
void scanInDevice(int n, int *devData) {
		int blockNum = (n + blockSize - 1) / blockSize;
		for (int d = 0; d < ilog2ceil(n) - 1; d++) {    //Here we have iterations
			upSweep << <blockNum, blockSize >> >(n, d, devData);  // Here we have kernal function
			checkCUDAError("upSweep not correct...");
		}
		//set last element to zero, refer to slides!
		int counter = 0;
		cudaMemcpy(&devData[n - 1], &counter, sizeof(int), cudaMemcpyHostToDevice);

		for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
			downSweep << <blockNum, blockSize >> >(n, d, devData);
			checkCUDAError("downSweep not correct...");
		}
	} 
```
```java
__global__ void upSweep(int N, int d, int *idata) {
	int n = (blockDim.x * blockIdx.x) + threadIdx.x;     
	if (n >= N) {
		return;
	}
	int delta = 1 << d;
	int doubleDelta = 1 << (d + 1);
	if (n % doubleDelta == 0) {    // not each thread is working, right? 
                                   //But those "should not be working" threads are still evoked via kernal function
		idata[n + doubleDelta - 1] += idata[n + delta - 1];
	}
	}
```

Plan to optimize this (yet several interviews this week I have to say: "lol" D:)    
Try to optimize mycode in path tracer project !

Also a huge problem: Memory I/O!    
We need to malloc memory in device, copy the host content into device then get a result, then transfer back to host memoy......  
When we're doing first assignment, we know that for index-continuous threads to access physical-not-continuous memory, it needs extra unnecessary operations and becomes slow.    
And in this project, we have a lot of memory I/O operation... So here we found another issue!!!
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/Profiling.PNG "Chart1")  
###
We can see CUDA memory operations occupied especially large part of the entire execution.
So my guess is right. :)

#### 5. Sample output 
More my test result are in the `result_showcase` folder.
Here I show one of them here:
####Array Size = 1 << 24. Block Size is 128.
```
****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  42   0 ]
==== cpu scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 411089014 411089056 ]
CPU scan power-of-two number time is 24.032100 ms
==== cpu scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 411088950 411088974 ]
    passed
CPU scan non-power-of-two number time is 42.639800 ms
==== naive scan, power-of-two ====
GPU Naive Scan time is 25.994272 ms
    passed
==== naive scan, non-power-of-two ====
GPU Naive Scan time is 25.812672 ms
    passed
==== work-efficient scan, power-of-two ====
GPU Efficient Scan time is 27.808865 ms
    passed
==== work-efficient scan, non-power-of-two ====
GPU Efficient Scan time is 24.770847 ms
    passed
==== thrust scan, power-of-two ====
GPU Thrust Scan time is 2.255712 ms
    passed
==== thrust scan, non-power-of-two ====
GPU Thrust Scan time is 2.052608 ms
    passed

*********************************************
*************** EXTRA CREDIT ****************
************* RADIX SORT TESTS **************
*************** POWER-OF-TWO ****************
*********************************************
    [  38 7719 21238 2437 8855 11797 8365 32285 10450 30612 5853 28100 1142 ... 7792 2304 ]
==== std sort for comparasion ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]

==== Extra : RadixSort ====
GPU Radix Sort time is 741.838257 ms
    passed
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]


*********************************************
*************** EXTRA CREDIT ****************
************* RADIX SORT TESTS **************
************* NON-POWER-OF-TWO **************
*********************************************
    [  38 7719 21238 2437 8855 11797 8365 32285 10450 30612 5853 28100 1142 ... 7792 2304 ]
==== std sort for comparasion ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]

==== Extra : RadixSort ====
GPU Radix Sort time is 743.634277 ms
    passed
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]


*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
CPU compact without scan power-of-two number time is 37.117100 ms
==== cpu compact without scan, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
CPU compact without scan non-power-of-two number time is 37.113600 ms
==== cpu compact with scan ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
CPU compact with scan time is 129.808400 ms
==== work-efficient compact, power-of-two ====
GPU Efficient Compact time is 27.158209 ms
    passed
==== work-efficient compact, non-power-of-two ====
GPU Efficient Compact time is 27.228865 ms
    passed
```
