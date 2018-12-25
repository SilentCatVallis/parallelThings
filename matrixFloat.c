#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mkl.h>

const int d1 = 1000;
const int d2 = 1000;
const float maxValue = 100;
const float minValue = -100;

struct dimension
{
	int D1;
	int D2;
};

void FillMatrix(struct dimension d, float* m)
{
    for (int i = 0; i < d.D1; i++)
	{
		for (int j = 0; j < d.D2; j++)
		{
			m[i*d.D2 + j] = (float)((maxValue - minValue) * ( (double)rand() / (double)RAND_MAX ) + minValue);
		}
	}
    return;
}

void PrintMaxtrixShort(float* m, struct dimension d)
{
    float result = 0;
	for (int i = 0; i < d.D1; i++)
	{
		for (int j = 0; j < d.D2; j++)
		{
			result += m[i*d.D2 + j];
		}
	}
	printf("Elements sum in the result matrix = %f\n", result);
}

int main (int argc, char *argv[]) {
	
    int m, n, k;
    
    //if (argc != 3)
    //{
    //    printf("error: use 'matrix.exe m n k' where m, n, k - matrices m*k and k*n size\n");
    //}
    if (sscanf (argv[1], "%d", &m) != 1) 
    {
        printf("error: use 'matrix.exe m n k' where m, n, k - matrices m*k and k*n size\n");
    }
    if (sscanf (argv[2], "%d", &n) != 1) 
    {
        printf("error: use 'matrix.exe m n k' where m, n, k - matrices m*k and k*n size\n");
    }
    if (sscanf (argv[3], "%d", &k) != 1) 
    {
        printf("error: use 'matrix.exe m n k' where m, n, k - matrices m*k and k*n size\n");
    }
    
    
	struct dimension matrix1Dimension;
	matrix1Dimension.D1 = m;
	matrix1Dimension.D2 = k;
	
	struct dimension matrix2Dimension;
	matrix2Dimension.D1 = k;
	matrix2Dimension.D2 = n;
	
	struct dimension resultDimension;
	resultDimension.D1 = matrix1Dimension.D1;
	resultDimension.D2 = matrix2Dimension.D2;
	
	
    srand (time(NULL));
    
    
    double time = omp_get_wtime();
		
    
    float *A, *B, *C;
    double alpha, beta;
    
    
    m = matrix1Dimension.D1;
    n = matrix2Dimension.D2;
    k = matrix1Dimension.D2;
    alpha = 1;
    beta = 0;    
    
    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
    
	FillMatrix(matrix1Dimension, A);
	FillMatrix(matrix2Dimension, B);
	
    time = omp_get_wtime() - time;
    printf("Generating matrices (with sizes m = %d, n = %d, k = %d) time: %f\n\n", m, n, k, time);
  
    
    
    
    
    int i = 0;
    
    time = omp_get_wtime();
    
	for (i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float sum = 0;  
			for (int l = 0; l < k; l++)
			{
				sum += A[i*k + l] * B[l*n + j];
			}
			
			C[i*n + j] = sum;
		}
	}
    
    time = omp_get_wtime() - time;
    printf("Single thread calculating time: %f\n", time);
	PrintMaxtrixShort(C, resultDimension);
	printf("\n");
    
    
    
    
    
    
    i = 0;
    
    time = omp_get_wtime();
    
    #pragma omp parallel for
	for (i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float sum = 0;  
			for (int l = 0; l < k; l++)
			{
				sum += A[i*k + l] * B[l*n + j];
			}
			
			C[i*n + j] = sum;
		}
	}
    
    time = omp_get_wtime() - time;
    printf("Parallel OMP calculating time (4 threads): %f\n", time);
	PrintMaxtrixShort(C, resultDimension);
	printf("\n");
    
    

    
    
    
    
    mkl_free(C);
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
    
    time = omp_get_wtime();
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
    
    time = omp_get_wtime() - time;
    printf("cblas_dgemm calculating time (4 threads): %f\n", time);
	PrintMaxtrixShort(C, resultDimension);
}