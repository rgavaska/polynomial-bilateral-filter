## Fast polynomial bilateral filter

This is a C++ implementation of the fast bilateral filtering algorithm proposed in the following paper:

R. G. Gavaskar and K. N. Chaudhury, "Fast Adaptive Bilateral Filtering", IEEE Transactions on Image Processing, vol. 28, no. 2, pp. 779-790, 2019.

DOI: 10.1109/TIP.2018.2871597

[[Paper]](https://ieeexplore.ieee.org/document/8469064)

[[arXiv]](https://arxiv.org/abs/1811.02308)

Note: This code only provides the implementation for the standard (not adaptive) bilateral filter using a box spatial kernel and a Gaussian range kernel.

The speedup comes from approximating local histograms at every pixel by a polynomial, as described in the paper. Users can specify the degree of the approximating polynomial at run-time. For purposes of numerical stability, it is recommended to not set the degree above 10. Good results can be obtained with the degree as low as 3 or 4.

### Requirements

(1) C++11 or higher with a supported compiler.

(2) OpenCV with C++ support.

This code has been tested using:

(1) G++ compiler version 9.3.0 for Ubuntu 20.04.

(2) OpenCV version 4.2.0. Might work for earlier versions as well.

### Usage

Compile the code using the command line as follows:
```
g++ main.cpp -o main.out `pkg-config --cflags --libs opencv4`
```
Run the code as follows:
```
./main.out <input_file> [sigma_s] [sigma_r] [N]
```
where
sigma_s = Half-width of box spatial filter,

sigma_r = Standard deviation of Gaussian range kernel,

N = Degree of polynomial.

Example:
```
./main.out peppers.png 4 50 2
```
