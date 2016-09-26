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
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/XueyinResultOriginal_pow(2%2C24).gif "Performance One") 


**FEATURES I IMPLEMENT**
-------------
####Part 1: CPU Scan & Stream Compaction
####Part 2: Naive GPU Scan Algorithm
####Part 3: Work-Efficient GPU Scan & Stream Compaction
####Part 4: Thrust Exclusive Scan using Thrust library
####Part 5: Radix Sort (Extra Credit)
####Part 6: Using std::chrono and CUDA events for comparing the speed of different algorithms      

**Dive Into Block Size**
-------------
In order to find the relationship between block size and performance, I modified block size to see different algorithm run time in order to get the optimized block size.   
Below is my chart based on my code:

**Case 1:** 
#####Power of  Two number, `SIZE` = 1 << 24 = 16777216
#####All the time recorded are in `ms`.

Block Size | Naïve Scan | Efficient Scan | Thrust Sort| Radix Sort|CPU Scan
---|---|---|---|---|---
16 | 52.045727 | 91.401695 |2.128768|2466.394043|24.0632
32 | 30.109312  | 53.902912 |2.09424|1302.962769|24.0563
64 | 25.546721  | 29.845119 |2.081152|786.153015|24.0908
128 | 25.994272 | 27.808865 |2.255712|741.838257|24.0321
256 | 25.615328 | 27.646433 |2.404192|769.176636|24.064
512 | 25.576256 | 29.840576 |2.256288|791.23053|24.5889
1024 | 25.609535 | 33.565887 |2.211232|860.626038|24.0653



**Case 2:**
#####Non Power of  Two number, `SIZE(NPOT)`= 1 << 24 - 3 = 16777213  
#####All the time recorded are in `ms`.
Block Size        | Naïve Scan   | Efficient Scan | Thrust Sort| Radix Sort|CPU Scan|
---|---|---|---|---|---
16 | 45.901855 | 89.639648 |2.094752|2448.440186|42.9234
32 | 30.138912  | 51.030048 |2.29776|1298.191284|43.1142
64 | 25.93968  | 27.795744 |2.011712|788.341675|42.6413
128 | 25.812672 | 24.770847 |2.052608|743.634277|42.6398
256 | 25.627424 | 27.607807 |2.223552|771.496643|41.6099
512 | 25.609535 | 29.848961 |2.146816|803.836487|42.1115
1024 | 26.082048 | 33.715874 |2.04576|851.824646|42.6002


Now let me draw a graph to explicitly show my result :)
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/ReadMeAboutBlockSizeChoose1.PNG "Chart1")  
###   
![alt text](https://github.com/xueyinw/Project2-Stream-Compaction/blob/master/result_showcase/ReadMeAboutBlockSizeChoose2.PNG "Chart2")

From case 1 and case 2, we could see that when block size is less than 128, the algorithm performance is definitely worse than block size = 128. And after we set block size to 128, we could see that radix sort performance reaches to the highest level. After block size continues to grow, we could notice that Naive Scan, Efficient Scan and Radix Sort are all becoming slower.
So I choose my block size to be `128` in my code.  


