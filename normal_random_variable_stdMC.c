#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "SFMT.h"

// N: sample size, S: dimension
// MAIN: number of independent runs
#define N (1024)
#define S (1024)
#define MAIN (25)

double** getmemory(int n,int m);

int main(void) {
    //initialization
    int i,j,k,m;
    int seed,seed_0 = 12345;
    int diff = 0;
    double *time_all;
    double **H,**tH,**C;
    clock_t start_all,end_all;
    double **A;
    double r;
    double **P;
    double **B;

    time_all = malloc(sizeof(double *) * MAIN);
    P = getmemory(N,S);
    B = getmemory(N,S);
    H = getmemory(S,S);
    tH = getmemory(S,S);
    C = getmemory(S,S);
    A = getmemory(S,S);

    //generate a random covariance matrix
    sfmt_t sfmt_0;
    sfmt_init_gen_rand(&sfmt_0, seed_0);
    for (i = 0; i < S; i++) {
        for (j = 0; j < S; j++) {
            H[i][j] = sfmt_genrand_real2(&sfmt_0);
            tH[j][i] = H[i][j];
        }
    }
    for (i = 0; i < S; i++) {
        for (j = 0; j < S; j++) {
            C[i][j] = 0;
            for (k = 0; k < S; k++) {
                C[i][j] += H[i][k] * tH[k][j];
            }
        }
    }

    //obtain the lower triangular matrix by Cholesky decomposition
    for (j = 0; j < S; j++) {
        r = C[j][j];
        for (k = 0; k < j; k++) {
            r -= A[j][k] * A[j][k];
        }
        if (r < 0) {
            fprintf(stderr,"r < 0\n");
            exit(0);
        }
        A[j][j] = sqrt(r);
        for (i = j + 1; i < S; i++) {
            r = C[i][j];
            for (k = 0; k < j; k++) {
                r -= A[i][k] * A[j][k];
            }
            A[i][j] = r / A[j][j];
        }
    }

    printf("%d\n",N);
    printf("%d\n",S);
    printf("\n");

    for (m = 0; m < MAIN; m++) {
        //set seed of random numbers
        sfmt_t sfmt;
        seed = (m + diff) * 1000;
        sfmt_init_gen_rand(&sfmt, seed);

        //start time measurement
        start_all = clock();

        //generate standard normally distributed vectors
        for (i = 0; i < N; i++) {
            for (j = 0; j < S; j++) {
                P[i][j] = sqrt(-2.0 * log(sfmt_genrand_real2(&sfmt))) * sin(2.0 * M_PI * sfmt_genrand_real2(&sfmt));
            }
        }

        //generate multivariate normal vectors (matrix-vector multiplication)
        for (i = 0; i < N; i++) {
            for (j = 0; j < S; j++) {
                B[i][j] = 0;
                for (k = 0; k < S; k++) {
                    B[i][j] += P[i][k] * A[k][j];
                }
            }
        }

        //stop time measurement
        end_all = clock();

        //get computational time
        time_all[m] = (double)(end_all-start_all)/CLOCKS_PER_SEC;
    }
    printf("\n");

    //get the average computational time
    double ave_time = 0;
    for (m = 0; m < MAIN; m++) {
        ave_time += time_all[m] / (double)MAIN;
    }
    printf("%.8f\n",ave_time);

    return 0;
}

//memory allocation
double** getmemory(int n,int m) {
    double** A;
    int i;
    A = malloc(sizeof(double *) * n);
    for (i = 0; i < n; i++) {
        A[i] = malloc(sizeof(double) * m);
    }
    return A;
}
