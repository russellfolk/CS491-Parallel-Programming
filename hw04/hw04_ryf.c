#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

#include "mpi.h"

#define N        101
#define MAX_ITER 1000000
#define EPSILON  0.0000001
#define NONE     0
#define BEGIN    1 // first message [send/receive original data]
#define COMPUTE  2 // computed values message [send/receive computed data]
#define DONE     3 // are we finished message
#define MASTER   0

double xold[(N+2)][(N+2)];
double xnew[(N+2)][(N+2)];

void initialize ();
double rtclock (void);

int main (int argc, char **argv)
{
	double clk_begin, clk_end, t;

	double this_diff, max_diff;
	int i, j, iter;

	int
		process_id, num_processes, // process id and max number of threads
		destination, source,       // used for message sending/receiving
		rows, offset, elements,    // used for sending rows
		num_rows, remaining_rows,  // used for dividing rows among threads
		msgtype,                   // storing the type of message for send/receive
		error = 1,                 // used if aborting threads...
		start, end;                // loop iteration for proportional usage

	MPI_Status status;

	double sum;

	/*  N is size of physical grid over which the heat equation is solved
		epsilon is the threshold for the convergence criterion
		xnew and xold hold the N*N new and old iterates for the temperature over grid
	*/

	// Initialization
	clk_begin = rtclock();

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	//printf("num_processes: %d\n", num_processes);
	num_rows = (int)N / (num_processes-1);
	remaining_rows = (int)N % (num_processes-1);

	initialize();

	int done = 0;
	// this is the section of code to optimize...
	for (iter = 0; iter < MAX_ITER; iter++)
	{
		if (process_id == MASTER)
		{
			offset = 0;
			// send data to the threads
			for (i = 1; i < num_processes; i++)
			{
				rows = num_rows+2; // need to send the row before and the row after
				if (i <= remaining_rows)
					rows++;
				elements = rows * (N+2); // there are N elements per row, and rows rows.
				destination = i;

				// send the offset so we know where we are in the array
				MPI_Send(&offset, 1, MPI_INT, destination, BEGIN, MPI_COMM_WORLD);
				// send the number of rows we're working with so we know how far to go
				MPI_Send(&rows, 1, MPI_INT, destination, BEGIN, MPI_COMM_WORLD);
				// send the data to compute with...
				MPI_Send(&xold[offset][0], elements, MPI_DOUBLE, destination, BEGIN, MPI_COMM_WORLD);
				offset += (rows-2); // need to send the next set with the row before and after
			}

			// wait for results
			for (i = 1; i < num_processes; i++)
			{
				source = i;
				// receive the offset so we know where we are in the array
				MPI_Recv(&offset, 1, MPI_INT, source, COMPUTE, MPI_COMM_WORLD, &status);
				// receive the number of rows we're working with and set elements for receive
				MPI_Recv(&rows, 1, MPI_INT, source, COMPUTE, MPI_COMM_WORLD, &status);
				elements = rows * (N+2);
				// receive the data
				MPI_Recv(&xnew[offset][0], elements, MPI_DOUBLE, source, COMPUTE, MPI_COMM_WORLD, &status);
				// receive the difference
				// this is unused due to a bug which due to time-constraints remains unsolved...
				MPI_Recv(&this_diff, 1, MPI_DOUBLE, source, COMPUTE, MPI_COMM_WORLD, &status);
				if (this_diff > max_diff)
				{
					max_diff = this_diff;
				}
			}

			max_diff = 0.0;
			for (i = 1; i < N+1; i++)
				for (j = 1; j < N+1; j++)
					if ((this_diff=fabs(xnew[i][j]-xold[i][j]))>max_diff)
						max_diff = this_diff;

			if (max_diff < EPSILON)
			{
				clk_end = rtclock();
				printf("Solution converged in  %d iterations with %d processes\n",iter+1, num_processes);
				printf("Solution at center of grid : %f\n",xnew[(N+1)/2][(N+1)/2]);
				t = clk_end - clk_begin;
				printf("Base-Jacobi: %.1f MFLOPS; Time = %.3f sec;\n",
				       4.0*N*N*(iter+1)/t/1000000, t);
				done = 1;
				for (i = 1; i < num_processes; i++)
					MPI_Send(&done, 1, MPI_INT, i, DONE, MPI_COMM_WORLD);
				break;
			}
			for (i = 1; i < num_processes; i++)
				MPI_Send(&done, 1, MPI_INT, i, DONE, MPI_COMM_WORLD);
			for (i = 1; i < N+1; i++)
				for (j = 1; j < N+1; j++)
				{
					xold[i][j] = xnew[i][j];
				}
		}
		else
		{
			// receive our offset so we know where we are in the array
			MPI_Recv(&offset, 1, MPI_INT, MASTER, BEGIN, MPI_COMM_WORLD, &status);
			// receive the number of rows we're working with so we know how far to go
			MPI_Recv(&rows, 1, MPI_INT, MASTER, BEGIN, MPI_COMM_WORLD, &status);
			// calculate the number of elements to receive
			elements = rows * (N+2);
			// receive the data to compute with...
			MPI_Recv(&xold[offset][0], elements, MPI_DOUBLE, MASTER, BEGIN, MPI_COMM_WORLD, &status);
			// calculate the results and the difference between the two...
			this_diff = 0.0;
			double thread_diff = 0.0;
			offset++;
			rows--;
			for (i = offset; i < offset+rows; i++)
			{
				for (j = 1; j < N+1; j++)
				{
					xnew[i][j] = 0.25 * (xold[i-1][j] + xold[i+1][j] + xold[i][j-1] + xold[i][j+1]);
					if ((this_diff = fabs(xnew[i][j]-xold[i][j])) > thread_diff)
						thread_diff = this_diff;
				}
			}
			// send back the data first send only the stuff changed...

			// send the offset so we know where we are in the array
			MPI_Send(&offset, 1, MPI_INT, MASTER, COMPUTE, MPI_COMM_WORLD);
			// send the number of rows we're working with so we know how far to go
			MPI_Send(&rows, 1, MPI_INT, MASTER, COMPUTE, MPI_COMM_WORLD);
			// calculate the number of elements to send
			elements = rows * (N+2);
			// send the data to compute with...
			MPI_Send(&xnew[offset][0], elements, MPI_DOUBLE, MASTER, COMPUTE, MPI_COMM_WORLD);
			// send back the diff
			MPI_Send(&thread_diff, 1, MPI_DOUBLE, MASTER, COMPUTE, MPI_COMM_WORLD);
			int done;
			MPI_Recv(&done, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD, &status);
			if (done == 1)
				break;
		}
	}
	MPI_Finalize();
	return 0;
}

void initialize ()
{
	int i, j;
	for (i=0; i<N+2; i++)
		xold[i][0] = i*50.0/(N+1);
	for (i=0; i<N+2; i++)
		xold[i][N+1] = (i+N+1)*50.0/(N+1);
	for (j=0; j<N+2; j++)
		xold[0][j] = j*50.0/(N+1);
	for (j=0; j<N+2; j++)
		xold[N+1][j] = (j+N+1)*50.0/(N+1);
	for (i=1; i<N+1;i++)
		for (j=1; j<N+1; j++)
			xold[i][j] = 0;
}

double rtclock (void)
{
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday (&Tp, &Tzp);
	if (stat != 0)
		printf("Error return from gettimeofday: %d",stat);
	return (Tp.tv_sec + Tp.tv_usec*1.0e-6);
}
