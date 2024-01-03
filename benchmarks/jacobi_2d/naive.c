#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define ASSERT 0

// grid has to be square
void init_arrays(int n, double A[n][n], double B[n][n])
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = ((double) i * (j + 2) + 2) / n; // TODO this is not randomly generated, so we might be optimising for this very specific case instead of generally -> consider
            B[i][j] = ((double) i * (j + 3) + 3) / n;
        }
    }
}

void print_arrays(int n, double A[n][n], double B[n][n])
{   
    puts("matrix A:");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", A[i][j]);
        }
        printf("\n");
    }
    
    puts("\nmatrix B:");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", B[i][j]);
        }
        printf("\n");
    }
}

void kernel_j2d(int tsteps, int n, double A[n][n], double B[n][n])
{
    // iterate for tsteps steps
    for (int t = 0; t < tsteps; t++)
    {
        for (int i = 1; i < (n - 1); i++)
        {
            for (int j = 1; j < (n - 1); j++)
            {
                B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]); // mean (sum/5) of -|- w/ Aij in centre
            }
        }
        for (int i = 1; i < (n - 1); i++)
        {
            for (int j = 1; j < (n - 1); j++)
            {
                A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j]);
            }
        }
    }
}

void run_bm(int tsteps, int n, const char* preset)
{    
    double (*A)[n][n] = malloc(sizeof(*A));
    double (*B)[n][n] = malloc(sizeof(*B));

    init_arrays(n, *A, *B);
    
    dphpc_time3(
        init_arrays(n, *A, *B),
        kernel_j2d(tsteps, n, *A, *B),
        preset
    );
    
    if (ASSERT && strcmp(preset, "S") == 0)
    {
        print_arrays(n, *A, *B);
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