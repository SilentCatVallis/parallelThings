#include <stdio.h>
#include <omp.h>

int main()
{
	const int N = 100000;
	int i, j;
	int num_prime;
	double time;
    int NumThreads;
    int threadId;
    int threadsCount;
    int localNumPrime;
    
	NumThreads = omp_get_num_procs();
    printf("Threads count: %d\n", NumThreads);
	
	num_prime = 0;
	time = omp_get_wtime();
    # pragma omp parallel shared(num_prime) private(threadId, threadsCount, localNumPrime, i, j)
    {
        threadId = omp_get_thread_num();
        threadsCount = omp_get_num_threads(); 
        localNumPrime = 0;
        
        for(i=2 + threadId; i<=N; i+= threadsCount)
        {
            int div = 0;
            for(j=2; j<=i; j++)
            {
                if(i % j == 0)
                {
                    div++;
                }
            }
            if(div == 1)
            {
                localNumPrime++;
            }
        }
        
        # pragma omp critical
        {
            num_prime += localNumPrime;
        }
    }
	time = omp_get_wtime() - time;
	
	printf("Prime numbers: %d\n", num_prime);
	printf("Time: %f\n", time);
	return 0;
}