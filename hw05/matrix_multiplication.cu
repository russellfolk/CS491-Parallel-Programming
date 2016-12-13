/**
  * Matrix Multiplication
  */

#include "matrix_multiplication.h"

int main (int argc, char **argv)
{
	double rtclock();
	double clkbegin, clkend, t;
	bool is_same;
	srand(time(NULL));
	extern char *optarg;
	extern int	 optind;

	int character, error = 0;

	char usage[] = "usage: %s [-N <size of matrix>]\n";

	while ((character = getopt(argc, argv, "N:?")) != -1)
		switch(character)
		{
			case 'N':
				matrix_size = atoi(optarg);
				break;
			case '?':
				error = 1;
				break;
		}

	if (error)
	{
		printf(usage, argv[0]);
		exit(error);
	}

	if (matrix_size < 1)
		matrix_size = DEFAULT_MATRIX_SIZE;

	printf("Matrix size is: %i x %i\n", matrix_size, matrix_size);

	// define the matrices to multiply
	int i, j, k;

	X      = (int *)malloc(matrix_size * matrix_size * sizeof(int*));
	Y      = (int *)malloc(matrix_size * matrix_size * sizeof(int*));
	Zcpu   = (int *)malloc(matrix_size * matrix_size * sizeof(int*));
	Zgpu   = (int *)malloc(matrix_size * matrix_size * sizeof(int*));
	Zgpu_s = (int *)malloc(matrix_size * matrix_size * sizeof(int*));

	for (i = 0; i < matrix_size * matrix_size; i++)
	{
		X[i]      = rand();
		Y[i]      = rand();
		Zcpu[i]   = 0;
		Zgpu[i]   = 0;
		Zgpu_s[i] = 0;
	}

	// start the clock: CPU
	clkbegin = rtclock();
	for (i = 0; i < matrix_size; i++)
		for (j = 0; j < matrix_size; j++)
			for (k = 0; k < matrix_size; k++)
				// multiply the matrix on the CPU
				Zcpu[matrix_size * i + j] +=
					X[matrix_size * i + k] * Y[matrix_size * k + j];
	// end the clock: CPU
	clkend = rtclock();
	t = clkend-clkbegin;

	printf("CPU matrix-multiplication finished.\n");
	printf("CPU Matrix-Multiplication: %.1f MFLOPS; Time = %.3f sec;\n",
						 4.0*matrix_size*matrix_size*matrix_size/t/1000000,t);

	// start the clock: GPU non-shared-memory
	clkbegin = rtclock();
	matrix_multiplication(X, Y, Zgpu);
	// end the clock: GPU non-shared-memory
	clkend = rtclock();
	t = clkend-clkbegin;

	printf("\nGPU non-shared-memory matrix-multiplication finished.\n");
	printf("GPU non-shared-memory Matrix-Multiplication: %.1f MFLOPS; Time = %.3f sec;\n",
						 4.0*matrix_size*matrix_size*matrix_size/t/1000000,t);

	// check if multiplication for GPU non-shared-memory matches CPU...
	is_same = true;
	for (i = 0; i < matrix_size; i++)
		for (j = 0; j < matrix_size; j++)
			if (Zcpu[matrix_size * i + j] != Zgpu[matrix_size * i + j])
				is_same = false;
	if (is_same)
		printf("GPU non-shared-memory Matrix-Multiplication completed successfully.\n");
	else
		printf("GPU non-shared-memory Matrix-Multiplication failed.\n");

	// start the clock: GPU shared-memory
	clkbegin = rtclock();
	matrix_multiplication(X, Y, Zgpu_s);
	// end the clock: GPU shared-memory
	clkend = rtclock();
	t = clkend-clkbegin;

	printf("\nGPU shared-memory matrix-multiplication finished.\n");
	printf("GPU shared-memory Matrix-Multiplication: %.1f MFLOPS; Time = %.3f sec;\n",
						 4.0*matrix_size*matrix_size*matrix_size/t/1000000,t);

	// check if multiplication for GPU shared-memory matches CPU...
	is_same = true;
	for (i = 0; i < matrix_size; i++)
		for (j = 0; j < matrix_size; j++)
			if (Zcpu[matrix_size * i + j] != Zgpu_s[matrix_size * i + j])
				is_same = false;
	if (is_same)
		printf("GPU shared-memory Matrix-Multiplication completed successfully.\n");
	else
		printf("GPU shared-memory Matrix-Multiplication failed.\n");

	return 0;
}

/**
  * Begin Non-Shared Memory Section
  *
  * This will take in two matrices (A, B) and produce C=A*B.
  */

void matrix_multiplication(int *A, int *B, int *C)
{
	int mem_size = matrix_size * matrix_size * sizeof(int);

	int *gpu_A, *gpu_B, *gpu_C;
	cudaError_t error_code;

	// Load matrix A into GPU device memory
	error_code = cudaMalloc(&gpu_A, mem_size);
	if (error_code != cudaSuccess)
		printf("CUDA malloc matrix A failed: %s\n",cudaGetErrorString(error_code));
	error_code = cudaMemcpy(gpu_A, A, mem_size, cudaMemcpyHostToDevice);
	if (error_code != cudaSuccess)
		printf("Copy matrix A to gpu device failed: %s\n",cudaGetErrorString(error_code));

	// Load matrix B into GPU device memory
	error_code = cudaMalloc(&gpu_B, mem_size);
	if (error_code != cudaSuccess)
		printf("CUDA malloc matrix B failed: %s\n",cudaGetErrorString(error_code));
	error_code = cudaMemcpy(gpu_B, B, mem_size, cudaMemcpyHostToDevice);
	if (error_code != cudaSuccess)
		printf("Copy matrix B to gpu device failed: %s\n",cudaGetErrorString(error_code));

	// Allocate matrix C into GPU device memory
	error_code = cudaMalloc(&gpu_C, mem_size);
	if (error_code != cudaSuccess)
		printf("CUDA malloc matrix C failed: %s\n",cudaGetErrorString(error_code));

	// Invoke the CUDA kernel to actually multiply the matrices
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((matrix_size + dimBlock.x - 1) / dimBlock.x,
		(matrix_size + dimBlock.y - 1) / dimBlock.y);
	matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(gpu_A, gpu_B, gpu_C, matrix_size);
	error_code = cudaThreadSynchronize();
	if (error_code != cudaSuccess)
		printf("Running CUDA kernel failed: %s\n", cudaGetErrorString(error_code));

	// Load matrix C from GPU device memory to CPU
	error_code = cudaMemcpy(C, gpu_C, mem_size, cudaMemcpyDeviceToHost);
	if (error_code != cudaSuccess)
		printf("Copy matrix C from gpu device failed: %s\n",cudaGetErrorString(error_code));

	// Free GPU device memory
	cudaFree(gpu_A);
	cudaFree(gpu_B);
	cudaFree(gpu_C);
}

/**
  * KERNEL: non-shared memory matrix multiplication
  *
  * Each thread computes one element of matrix C by multiplying a single row from matrix A
  * with a single column from matrix B and accumulating the results into the value c.
  */
__global__ void matrix_multiplication_kernel(int *A, int *B, int *C, int N)
{
	int c = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > N || col > N)
		return;
	for (int i = 0; i < N; i++)
			c += (A[row * N + i]) * (B[i * N + col]);

	C[row * N + col] = c;
}


/**
  * Begin Shared Memory Section
  *
  * This will take in two matrices (A, B) and produce C=A*B.
  */

void matrix_multiplication_shared(int *A, int *B, int *C)
{
	int mem_size = matrix_size * matrix_size * sizeof(int);

	int *gpu_A, *gpu_B, *gpu_C;
	cudaError_t error_code;

	// Load matrix A into GPU device memory
	error_code = cudaMalloc(&gpu_A, mem_size);
	if (error_code != cudaSuccess)
		printf("CUDA malloc matrix A failed: %s\n",cudaGetErrorString(error_code));
	error_code = cudaMemcpy(gpu_A, A, mem_size, cudaMemcpyHostToDevice);
	if (error_code != cudaSuccess)
		printf("Copy matrix A to gpu device failed: %s\n",cudaGetErrorString(error_code));

	// Load matrix B into GPU device memory
	error_code = cudaMalloc(&gpu_B, mem_size);
	if (error_code != cudaSuccess)
		printf("CUDA malloc matrix B failed: %s\n",cudaGetErrorString(error_code));
	error_code = cudaMemcpy(gpu_B, B, mem_size, cudaMemcpyHostToDevice);
	if (error_code != cudaSuccess)
		printf("Copy matrix B to gpu device failed: %s\n",cudaGetErrorString(error_code));

	// Allocate matrix C into GPU device memory
	error_code = cudaMalloc(&gpu_C, mem_size);
	if (error_code != cudaSuccess)
		printf("CUDA malloc matrix C failed: %s\n",cudaGetErrorString(error_code));

	// Invoke the CUDA kernel to actually multiply the matrices
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(matrix_size / dimBlock.x, matrix_size / dimBlock.y);
	matrix_mult_shared_kernel<<<dimGrid, dimBlock>>>(gpu_A, gpu_B, gpu_C, matrix_size);
	error_code = cudaThreadSynchronize();
	if (error_code != cudaSuccess)
		printf("Running CUDA kernel failed: %s\n", cudaGetErrorString(error_code));

	// Load matrix C from GPU device memory to CPU
	error_code = cudaMemcpy(C, gpu_C, mem_size, cudaMemcpyDeviceToHost);
	if (error_code != cudaSuccess)
		printf("Copy matrix C from gpu device failed: %s\n",cudaGetErrorString(error_code));

	// Free GPU device memory
	cudaFree(gpu_A);
	cudaFree(gpu_B);
	cudaFree(gpu_C);
}

/**
  * KERNEL: shared-memory matrix multiplication
  *
  * Each thread computes one element of a sub-matrix of matrix C by multiplying a single
  * row from a sub-matrix of matrix A with a single column from a sub-matrix of matrix B
  * and accumulating the results into sub-value for the sub-matrix of matrix C. These
  * values are then accumulated into the resulting matrix C.
  */
__global__ void matrix_mult_shared_kernel(int *A, int *B, int *C, int N)
{
	// Block row and column
	int block_row = blockIdx.y;
	int block_col = blockIdx.x;

	int *C_sub = get_sub_matrix(C, N, block_row, block_col);

	int c = 0;

	// Thread row and column within C_sub
	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int i = 0; i < (N / BLOCK_SIZE); i++)
	{
		int *A_sub = get_sub_matrix(A, N, block_row, i);
		int *B_sub = get_sub_matrix(B, N, i, block_col);

		// Shared memory used to store A_sub and B_sub respectively
		__shared__ int A_shared[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int B_shared[BLOCK_SIZE][BLOCK_SIZE];

		// Each thread loads one element of each sub-matrix to shared memory
		A_shared[row][col] = get_matrix_element(A_sub, N, row, col);
		B_shared[row][col] = get_matrix_element(B_sub, N, row, col);

		// need to be in sync before continuing
		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; j++)
			c += A_shared[row][j] * B_shared[j][col];

		// need to be in sync before continuing
		__syncthreads();
	}

	set_matrix_element(C_sub, N, row, col, c);
}

// Get the BLOCK_SIZE * BLOCK_SIZE sub-matrix of a given matrix,
// needed for shared memory implementation
__device__ int* get_sub_matrix(int *matrix, int N, int row, int col)
{
	int *matrix_sub;
	matrix_sub = &matrix[N * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return matrix_sub;
}

// Get a single matrix element, needed for shared memory implementation
__device__ int get_matrix_element(int* A, int N, int row, int col)
{
	return A[row * N + col];
}

// Set a single matrix element, needed for shared memory implementation
__device__ void set_matrix_element(int *A, int N, int row, int col, int value)
{
	A[row * N + col] = value;
}
