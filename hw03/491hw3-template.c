#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define N 101
#define maxiter 1000000
#define epsilon 0.0000001

double xold[(N+2)][(N+2)];
double xnew[(N+2)][(N+2)];

main (int argc, char * argv[])
{
	double rtclock();
	double clkbegin, clkend, t;

	double thisdiff, maxdiff;
	int i, j, iter;

	/*  N is size of physical grid over which the heat equation is solved
		epsilon is the threshold for the convergence criterion
		xnew and xold hold the N*N new and old iterates for the temperature over grid
	*/

	// Initialization

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

	clkbegin = rtclock();

	// this is the section of code to optimize...
	for (iter=0; iter<maxiter; iter++)
	{
		maxdiff = 0;
		for (i=1; i<N+1; i++)
			for (j=1; j<N+1; j++)
			{
				xnew[i][j] = 0.25*(xold[i-1][j]+xold[i+1][j]+xold[i][j-1]+xold[i][j+1]);
				if ((thisdiff=fabs(xnew[i][j]-xold[i][j]))>maxdiff)
					maxdiff = thisdiff;
			}
		if(maxdiff<epsilon)
		{
			clkend = rtclock();
			printf("Solution converged in  %d iterations\n",iter+1);
			printf("Solution at center of grid : %f\n",xnew[(N+1)/2][(N+1)/2]);
			t = clkend-clkbegin;
			printf("Base-Jacobi: %.1f MFLOPS; Time = %.3f sec; \n",
			       4.0*N*N*(iter+1)/t/1000000,t);
			break;
		}

		for (i=1; i<N+1; i++)
			for (j=1; j<N+1; j++)
				xold[i][j] = xnew[i][j];
	}
}

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

