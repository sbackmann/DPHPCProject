#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define ASSERT 0

// grid has to be square
void init_arrays(int n, double *A, double *B)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = ((double) i * (j + 2) + 2) / n;
            B[i * n + j] = ((double) i * (j + 3) + 3) / n;
        }
    }
    
    
    // for (int j = 0; j < n; j++)
    // {
    //     B[j] = ((double) 3) / n;
    // }
    // for (int i = 1; i < (n - 1); i++)
    // {
    //     B[i * n] = ((double) 3 * i + 3) / n;
    //     B[i * n + (n - 1)] = ((double) (n + 2) * i + 3) / n;
    // }
    // for (int j = 0; j < n; j++)
    // {
    //     B[(n - 1) * n + j] = ((double) (n - 1) * j + n * 3) / n;
    // }
}

void print_arrays(int n, double *A, double *B)
{   
    puts("matrix A:");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", A[i * n + j]);
        }
        printf("\n");
    }
    
    puts("\nmatrix B:");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", B[i * n + j]);
        }
        printf("\n");
    }
}

void kernel_j2d(int tsteps, int n, double *A, double *B)
{
    // iterate for tsteps steps
    for (int t = 0; t < tsteps; t++)
    {
        // DEFAULT
        // for (int i = 1; i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + (j - 1)] + A[i * n + (j + 1)] + A[(i + 1) * n + j] + A[(i - 1) * n + j]); // mean (sum/5) of -|- w/ Aij in centre
        //     }
        // }
        // for (int i = 1; i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         A[i * n + j] = 0.2 * (B[i * n + j] + B[i * n + (j - 1)] + B[i * n + (j + 1)] + B[(i + 1) * n + j] + B[(i - 1) * n + j]);
        //     }
        // }
        
        // BETTER LOCALITY
        // for (int i = 1; i < 3; i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
        //     }
        // }
        // for (int i = 3; i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
        //         A[(i-2) * n + j] = 0.2 * (B[(i-2) * n + j] + B[(i-2) * n + j-1] + B[(i-2) * n + j+1] + B[(i-1) * n + j] + B[(i-3) * n + j]);
        //     }
        // }
        // for (int i = (n - 3); i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         A[i * n + j] = 0.2 * (B[i * n + j] + B[i * n + j-1] + B[i * n + j+1] + B[(i+1) * n + j] + B[(i-1) * n + j]);
        //     }
        // }
        
        // UNROLL 2X
        // int i;
        // for (i = 1; i < 3; i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
        //     }
        // }
        // for (i = 3; i < (n - 2); i+=2)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
        //         B[(i+1) * n + j] = 0.2 * (A[(i+1) * n + j] + A[(i+1) * n + j-1] + A[(i+1) * n + j+1] + A[(i+2) * n + j] + A[i * n + j]);
                
        //         A[(i-2) * n + j] = 0.2 * (B[(i-2) * n + j] + B[(i-2) * n + j-1] + B[(i-2) * n + j+1] + B[(i-1) * n + j] + B[(i-3) * n + j]);
        //         A[(i-1) * n + j] = 0.2 * (B[(i-1) * n + j] + B[(i-1) * n + j-1] + B[(i-1) * n + j+1] + B[i * n + j] + B[(i-2) * n + j]);                
        //     }
        // }
        // for (; i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
        //         A[(i-2) * n + j] = 0.2 * (B[(i-2) * n + j] + B[(i-2) * n + j-1] + B[(i-2) * n + j+1] + B[(i-1) * n + j] + B[(i-3) * n + j]);
        //     }
        // }
        // for (i = (n - 3); i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         A[i * n + j] = 0.2 * (B[i * n + j] + B[i * n + j-1] + B[i * n + j+1] + B[(i+1) * n + j] + B[(i-1) * n + j]);
        //     }
        // }
        
        // UNROLL OUTER 2X, INNER 2X
        int i, j;
        for (i = 1; i < 3; i++)
        {
            for (j = 1; j < (n - 1); j++)
            {
                B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
            }
        }
        for (i = 3; i < (n - 2); i+=2)
        {
            for (j = 1; j < (n - 2); j+=2)
            {
                B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);    // 1
                B[i * n + j+1] = 0.2 * (A[i * n + j+1] + A[i * n + j] + A[i * n + j+2] + A[(i+1) * n + j+1] + A[(i-1) * n + j+1]);  // 2
                B[(i+1) * n + j] = 0.2 * (A[(i+1) * n + j] + A[(i+1) * n + j-1] + A[(i+1) * n + j+1] + A[(i+2) * n + j] + A[i * n + j]);
                B[(i+1) * n + j+1] = 0.2 * (A[(i+1) * n + j+1] + A[(i+1) * n + j] + A[(i+1) * n + j+2] + A[(i+2) * n + j+1] + A[i * n + j+1]);
                
                A[(i-2) * n + j] = 0.2 * (B[(i-2) * n + j] + B[(i-2) * n + j-1] + B[(i-2) * n + j+1] + B[(i-1) * n + j] + B[(i-3) * n + j]);
                A[(i-2) * n + j+1] = 0.2 * (B[(i-2) * n + j+1] + B[(i-2) * n + j] + B[(i-2) * n + j+2] + B[(i-1) * n + j+1] + B[(i-3) * n + j+1]);
                A[(i-1) * n + j] = 0.2 * (B[(i-1) * n + j] + B[(i-1) * n + j-1] + B[(i-1) * n + j+1] + B[i * n + j] + B[(i-2) * n + j]); // dep on 1
                A[(i-1) * n + j+1] = 0.2 * (B[(i-1) * n + j+1] + B[(i-1) * n + j] + B[(i-1) * n + j+2] + B[i * n + j+1] + B[(i-2) * n + j+1]); // dep on 2             
            }
            
            for (; j < (n - 1); j++)
            {
                B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
                B[(i+1) * n + j] = 0.2 * (A[(i+1) * n + j] + A[(i+1) * n + j-1] + A[(i+1) * n + j+1] + A[(i+2) * n + j] + A[i * n + j]);
                
                A[(i-2) * n + j] = 0.2 * (B[(i-2) * n + j] + B[(i-2) * n + j-1] + B[(i-2) * n + j+1] + B[(i-1) * n + j] + B[(i-3) * n + j]);
                A[(i-1) * n + j] = 0.2 * (B[(i-1) * n + j] + B[(i-1) * n + j-1] + B[(i-1) * n + j+1] + B[i * n + j] + B[(i-2) * n + j]);                
            }
        }
        for (; i < (n - 1); i++)
        {
            for (j = 1; j < (n - 1); j++)
            {
                B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + j-1] + A[i * n + j+1] + A[(i+1) * n + j] + A[(i-1) * n + j]);
                A[(i-2) * n + j] = 0.2 * (B[(i-2) * n + j] + B[(i-2) * n + j-1] + B[(i-2) * n + j+1] + B[(i-1) * n + j] + B[(i-3) * n + j]);
            }
        }
        for (i = (n - 3); i < (n - 1); i++)
        {
            for (j = 1; j < (n - 1); j++)
            {
                A[i * n + j] = 0.2 * (B[i * n + j] + B[i * n + j-1] + B[i * n + j+1] + B[(i+1) * n + j] + B[(i-1) * n + j]);
            }
        }
    }
}

void run_bm(int tsteps, int n, const char* preset)
{    
    // for certain optimisations, it's more performant to have an 1D array like double *A = malloc(sizeof(*A) * n * n), access by A[i * n + j]
    double *A = malloc(sizeof(*A) * n * n);
    double *B = malloc(sizeof(*B) * n * n);

    init_arrays(n, A, B);
    
    dphpc_time3(
        init_arrays(n, A, B),
        kernel_j2d(tsteps, n, A, B),
        preset
    );
    
    if (ASSERT && strcmp(preset, "S") == 0)
    {
        print_arrays(n, A, B);
    }

    free((void *) A);
    free((void *) B);
}

int main(int argc, char** argv)
{
    run_bm(50, 150, "S");   // steps 50, n 150
    run_bm(80, 350, "M");   // steps 80, n 350
    run_bm(200, 700, "L");   // steps 200, n 700
    //run_bm(500, 1400, "paper"); // in-between for testing
    run_bm(1000, 2800, "paper");  // steps 1000, n 2800
  
    return 0;
}