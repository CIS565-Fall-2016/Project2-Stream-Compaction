CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yaoyi Bai
* Tested on: Windows 7, i5-3210 @ 2.50GHz 6GB, GTX 640M 2GB (personal machine)

### (TODO: Yaoyi Bai)

Analysis
The main target of the project is removing 0s from 1s using different calculation methods. We should implement CPU methods, naive sorting methods and efficient sorting methods. 
All of calculation methods are conducted in power-of-two version and also non-power-of-two version. Then we test all of these methods using different block sizes. 
These sizes are 8, 16, 64, 128 and 256. 
According to the result of rumtime shown in image_7.jpg, it would be easy to find out that the cpu non-power-of-two calculation is the most inefficient method, which will take hundreds of times more time to calculate the result. Since it can only calculate the result using for loop. 
Then the non-power-of-two CPU calculation method is faster because of the simplification of the calculation method.
As for the GPU calculation methods, we applied multi-thread to the calculation so that we can perform at least 8 calculation as well. Especially the efficient calculation method. By applying the down-loop and up-loop method, we can avoid for loop in CPU and also avoid the data transformation between CPU and GPU which will take more time than the calculation inside the ALUs in CPU or GPU. 
The target of the project is to speed up the calculation, there are several ways for us to perform the calculation. First, we can use parallel thread in side GPU kernel. Second, we should figure out ways to shrink the transformation time between GPU and CPU, especially when the data is far more greater than we are testing. We can see that the bottleneck of the project lie in that we should frequently transfer data from CPU to GPU, and this is where efficient method can speed up the naive method to some extent. 

