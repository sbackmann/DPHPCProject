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
            A[i][j] = ((double) i * (j + 2) + 2) / n;
            B[i][j] = ((double) i * (j + 3) + 3) / n;
        }
    }
    
    
    // for (int j = 0; j < n; j++)
    // {
    //     B[0][j] = ((double) 3) / n;
    // }
    // for (int i = 1; i < (n - 1); i++)
    // {
    //     B[i][0] = ((double) 3 * i + 3) / n;
    //     B[i][n-1] = ((double) (n + 2) * i + 3) / n;
    // }
    // for (int j = 0; j < n; j++)
    // {
    //     B[n-1][j] = ((double) (n - 1) * j + n * 3) / n;
    // }
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
        // BETTER LOCALITY
        // for (int i = 1; i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
        //         if (i > 2)
        //         {
        //             A[i-2][j] = 0.2 * (B[i-2][j] + B[i-2][j-1] + B[i-2][j+1] + B[i-1][j] + B[i-3][j]);
        //         }
        //     }
        // }
        // for (int i = 0; i < 2; i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         A[n-3+i][j] = 0.2 * (B[n-3+i][j] + B[n-3+i][j-1] + B[n-3+i][j+1] + B[n-2+i][j] + B[n-4+i][j]);
        //     }
        // }
        
        // NO BRANCHING
        // for (int i = 1; i < 3; i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
        //     }
        // }
        // for (int i = 3; i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
        //         A[i-2][j] = 0.2 * (B[i-2][j] + B[i-2][j-1] + B[i-2][j+1] + B[i-1][j] + B[i-3][j]);
        //     }
        // }
        // for (int i = (n - 3); i < (n - 1); i++)
        // {
        //     for (int j = 1; j < (n - 1); j++)
        //     {
        //         A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j]);
        //     }
        // }
        
        // UNROLL OUTER 2X
        int i;
        for (i = 1; i < 3; i++)
        {
            for (int j = 1; j < (n - 1); j++)
            {
                B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
            }
        }
        for (i = 3; i < (n - 2); i+=2)
        {
            for (int j = 1; j < (n - 1); j++)
            {
                B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
                B[i+1][j] = 0.2 * (A[i+1][j] + A[i+1][j-1] + A[i+1][j+1] + A[i+2][j] + A[i][j]);
                
                A[i-2][j] = 0.2 * (B[i-2][j] + B[i-2][j-1] + B[i-2][j+1] + B[i-1][j] + B[i-3][j]);
                A[i-1][j] = 0.2 * (B[i-1][j] + B[i-1][j-1] + B[i-1][j+1] + B[i][j] + B[i-2][j]);
            }
        }
        for (; i < (n - 1); i++)
        {
            for (int j = 1; j < (n - 1); j++)
            {
                B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
                A[i-2][j] = 0.2 * (B[i-2][j] + B[i-2][j-1] + B[i-2][j+1] + B[i-1][j] + B[i-3][j]);
            }
        }
        for (i = (n - 3); i < (n - 1); i++)
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

#include "_parameters.h"
int main(int argc, char** argv)
{   
    const char *presets[] = {"S", "M", "L", "paper"};

    for (int i = 0; i < 4; i++) {
        const char* preset = presets[i];
        int tsteps = get_params(preset)[0];
        int n      = get_params(preset)[1];
        run_bm(tsteps, n, preset);
    }


    //run_bm(500, 1400, "missing"); // in-between for testing
  
    return 0;
}