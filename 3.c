#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x)
{
    return sin(x);
}

int main()
{
    const long int N = 1000*1000*1000;
    double sum;
    double a = 0.0;
    double b = 1.0;
    double h = (b - a) / N;
    double time;
    int NumThreads = omp_get_num_procs();
    
    printf("Threads count: %d\n", NumThreads);

    long int i;
    time = omp_get_wtime();
    sum = 0.0;
    for(i=0; i<N; i++)
    {
		sum += f(a + (i + 0.5)*h)*h;
    }
    time = omp_get_wtime() - time;
    printf("Serial time: %f\n", time);
    printf("Serial sum: %f\n", sum);
    
    time = omp_get_wtime();
    sum = 0.0;
    
    long int firstIndex;
    long int lastIndex;
    long int threadId;
    long int threadsCount;
    double localSum;
    
	//TODO: why does it work incorrectly?
    #pragma omp parallel num_threads(NumThreads) shared(sum) private(firstIndex, lastIndex, threadId, threadsCount, localSum, i)
    {
        threadId = omp_get_thread_num();
        threadsCount = omp_get_num_threads(); 
        localSum = 0;
        
        for(i=threadId; i<N; i+=threadsCount)
        {
            localSum += f(a + (i + 0.5)*h)*h;
        }
        
        #pragma omp critical
        {
            sum += localSum;
        }
    }
    time = omp_get_wtime() - time;
    printf("Parallel time: %f\n", time);
    printf("Parallel sum: %f\n", sum);
    
    return 0;
}