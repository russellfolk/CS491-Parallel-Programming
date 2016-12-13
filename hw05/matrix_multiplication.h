#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

// default Matrix Size (N * N) where N = 1024
#define DEFAULT_MATRIX_SIZE 1024

// thread block size, defined to the default of the specs
#define BLOCK_SIZE 32

// size of matrix
unsigned int matrix_size = 0;

int *X;
int *Y;
int *Zcpu;
int *Zgpu;
int *Zgpu_s;

double rtclock()
{
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday (&Tp, &Tzp);
	if (stat != 0)
		printf("Error return from gettimeofday: %d",stat);
	return (Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

// non-shared components
void matrix_multiplication(int *, int *, int *);
__global__ void matrix_multiplication_kernel(int *, int *, int *, int);

// shared memory components
void matrix_multiplication_shared(int *, int *, int *);
__global__ void matrix_mult_shared_kernel(int *, int *, int *, int);
__device__ int* get_sub_matrix(int *, int, int, int);
__device__ int get_matrix_element(int*, int, int, int);
__device__ void set_matrix_element(int *, int, int, int, int);
