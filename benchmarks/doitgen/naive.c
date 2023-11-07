/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* doitgen.c: this file is part of PolyBench/C */

#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>

#define ASSERT 0

static
void init_array(int nr, int nq, int np,
  double A[nr][nq][np],
  double C4[np][np])
{

    for (int i = 0; i < nr; i++)
        for (int j = 0; j < nq; j++)
            for (int k = 0; k < np; k++)
                A[i][j][k] = (double) ((i*j + k)%np) / np;
    for (int i = 0; i < np; i++)
        for (int j = 0; j < np; j++)
            C4[i][j] = (double) (i*j % np) / np;
}


static
void print_array(int nr, int nq, int np,
   double A[nr][nq][np])
{
    
    int i, j, k;

    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "A");
    for (i = 0; i < nr; i++)
        for (j = 0; j < nq; j++)
            for (k = 0; k < np; k++) {
                if ((i*nq*np+j*np+k) % 20 == 0) fprintf (stderr, "\n");
                fprintf (stderr, "%0.2lf ", A[i][j][k]);
        }
    fprintf(stderr, "\nend   dump: %s\n", "A");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}


void kernel_doitgen(int nr, int nq, int np,
      double A[nr][nq][np],
      double C4[np][np],
      double sum[np])
{

    #pragma scop
    for (int r = 0; r < nr; r++) {
        for (int q = 0; q < nq; q++) {
            for (int p = 0; p < np; p++) {
                sum[p] = 0.0;
                for (int s = 0; s < np; s++)
                sum[p] = sum[p] + A[r][q][s] * C4[s][p];
            }
            for (int p = 0; p < np; p++)
                A[r][q][p] = sum[p];
        }
    }
    #pragma endscop

}


void serializeArray(int nr, int nq, int np,
    double A[nr][nq][np], const char *prefix) {
    char path[100]; 
    snprintf(path, sizeof(path), "benchmarks/doitgen/test_cases/%s.dat", prefix);

    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    if (fwrite(A, sizeof(double), nr * nq * np, file) != nr * nq * np) {
        printf("Error writing to file.\n");
        exit(1);
    }

    fclose(file);
    printf("Array serialized and written to file.\n");
}


void assertCorrectness(int nr, int nq, int np,
    double A[nr][nq][np], const char *prefix) {

    // Somehow only works when a.exe run manually
    // When path is amended to "test_cases/%s.dat", no output is produced

    fprintf(stderr, "A[0][0][1] = %f\n", A[0][0][1]);
    char path[100];
    snprintf(path, sizeof(path), "benchmarks/doitgen/test_cases/%s.dat", prefix);
    printf("Reading from %s\n", path);
    FILE *file = fopen(path, "rb");
    printf("Array deserialized.\n");
    if (!file)
        perror("fopen");

    double A_test[nr][nq][np];

    if (fread(A_test, sizeof(double), nr * nq * np, file) != nr * nq * np) {
        printf("Error reading file.\n");
        exit(1);
    }

    fclose(file);
    // Perform assertion
    for (int r = 0; r < nr; r++) {
        for (int q = 0; q < nq; q++) {
            for (int p = 0; p < np; p++){
                if (A[r][q][p] != A_test[r][q][p]) {
                    printf("A[%d][%d][%d] = %f\n", r, q, p, A[r][q][p]);
                    printf("A_test[%d][%d][%d] = %f\n", r, q, p, A_test[r][q][p]);
                    printf("Arrays are not equal.\n");
                    exit(1);
                }
            }
        }
    }
    printf("Arrays are equal.\n");
}


void run_bm(int nr, int nq, int np, const char* preset) {
    
    double (*A)[nr][nq][np]; A = (double(*)[nr][nq][np])malloc ((nr) * (nq) * (np)* sizeof(double));;
    double (*sum)[np]; sum = (double(*)[np])malloc ((np)* sizeof(double));;
    double (*C4)[np][np]; C4 = (double(*)[np][np])malloc ((np) * (np)* sizeof(double));;

    init_array (nr, nq, np,
        *A,
        *C4);
    
    dphpc_time3(
        init_array(nr, nq, np, *A, *C4),
        kernel_doitgen(nr, nq, np, *A, *C4, *sum),
        preset
    );
    if (ASSERT && strcmp(preset, "S") == 0) {
        assertCorrectness(nr, nq, np, *A, preset);
    }
    free((void*)A);;
    free((void*)sum);;
    free((void*)C4);;
}

int main(int argc, char** argv)
{

    int nr = 60;
    int nq = 60;
    int np = 128;
    run_bm(nr, nq, np, "S");
    
    nr = 110;
    nq = 125;
    np = 256;
    run_bm(nr, nq, np, "M");

    nr = 220;
    nq = 250;
    np = 512;
    run_bm(nr, nq, np, "L");

    nr = 220;
    nq = 250;
    np = 270;
    run_bm(nr, nq, np, "paper");

    return 0;
}
