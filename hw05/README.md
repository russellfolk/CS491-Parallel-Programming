Russell Folk
C S 521 Programming
Homework 5

# CUDA Practice Project
## Matrix Multiplication

Required files are
* matrix_multiplication.cu
* matrix_multiplication.h

To compile the program
* `nvcc -o matrix matrix_multiplication.cu`

To run the program
* `./matrix [-N <size of matrix>]`
* N flag is optional
* N defaults to 1024

### Results
Matrix size is: 512
CPU multiplication finished.
CPU Matrix-Multiplication: 590.5 MFLOPS; Time = 0.909 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 6463.0 MFLOPS; Time = 0.083 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 17457.0 MFLOPS; Time = 0.031 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 512
CPU multiplication finished.
CPU Matrix-Multiplication: 611.8 MFLOPS; Time = 0.878 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 9004.7 MFLOPS; Time = 0.060 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 17429.7 MFLOPS; Time = 0.031 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 512
CPU multiplication finished.
CPU Matrix-Multiplication: 610.1 MFLOPS; Time = 0.880 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 9465.3 MFLOPS; Time = 0.057 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 17479.7 MFLOPS; Time = 0.031 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 1024
CPU multiplication finished.
CPU Matrix-Multiplication: 634.4 MFLOPS; Time = 6.770 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 15555.7 MFLOPS; Time = 0.276 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 17215.1 MFLOPS; Time = 0.249 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 1024
CPU multiplication finished.
CPU Matrix-Multiplication: 639.5 MFLOPS; Time = 6.716 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 15681.4 MFLOPS; Time = 0.274 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 17214.2 MFLOPS; Time = 0.250 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 1024
CPU multiplication finished.
CPU Matrix-Multiplication: 629.2 MFLOPS; Time = 6.826 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 15795.7 MFLOPS; Time = 0.272 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 17217.7 MFLOPS; Time = 0.249 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 2048
CPU multiplication finished.
CPU Matrix-Multiplication: 181.3 MFLOPS; Time = 189.514 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 19068.1 MFLOPS; Time = 1.802 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 19649.6 MFLOPS; Time = 1.749 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 2048
CPU multiplication finished.
CPU Matrix-Multiplication: 182.9 MFLOPS; Time = 187.859 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 19174.5 MFLOPS; Time = 1.792 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 19630.6 MFLOPS; Time = 1.750 sec;
GPU shared-memory Matrix-Multiplication completed successfully.

Matrix size is: 2048
CPU multiplication finished.
CPU Matrix-Multiplication: 180.9 MFLOPS; Time = 189.927 sec;

GPU non-shared-memory multiplication finished.
GPU non-shared-memory Matrix-Multiplication: 19045.2 MFLOPS; Time = 1.804 sec;
GPU non-shared-memory Matrix-Multiplication completed successfully.

GPU shared-memory multiplication finished.
GPU shared-memory Matrix-Multiplication: 19646.2 MFLOPS; Time = 1.749 sec;
GPU shared-memory Matrix-Multiplication completed successfully.