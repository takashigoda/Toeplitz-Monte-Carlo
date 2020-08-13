#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "SFMT.h"
#include "fftw3.h"

// N: sample size, S: dimension, M: number of finite elements
// MAIN: number of independent runs
// C:coefficient appearing in the random field a
#define N (64)
#define M (8)
#define S (64)
#define C (1)
#define MAIN (25)

double** getmemory2(int n,int m);
double*** getmemory3(int n,int m,int l);
double**** getmemory4(int o,int n,int m,int l);
double psi(int j,double X_i);
double*** LU(int n,double **A);
double* linear_system(int n,double **A,double *b);

int main(void) {
    //initialization
    int m,i,j,k,l,n;
    int condition = 0;
    int seed;
    double X = 0.5;
    int diff = 0;
    double alpha = 2;
    double *time_all,*u_x;
    clock_t start_all,end_all;
    double **Y;
    double **psi_kl;
    double ****theta;
    double ***B;
    double *g_hat,**u_hat;
    double *u;
    double u_est = 0;

    time_all = malloc(sizeof(double *) * MAIN);
    u_x = malloc(sizeof(double *) * MAIN);
    Y = getmemory2(N,S);
    psi_kl = getmemory2(S,3);
    theta = getmemory4(3,N,M - 1,3);
    B = getmemory3(N,M - 1,3);
    g_hat = malloc(sizeof(double *) * (M - 1));
    u_hat = getmemory2(N,M - 1);
    u = malloc(sizeof(double *) * N);

    printf("%d\n",N);
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
              if (condition == 0) {
                  Y[i][j] = sqrt(-2.0 * log(sfmt_genrand_real2(&sfmt))) * sin(2.0 * M_PI * sfmt_genrand_real2(&sfmt));
              }
              else {
                  Y[i][j] = sfmt_genrand_real2(&sfmt) - 0.5;
              }
            }
        }

        //compute the stiff matrix B
        for (n = 0; n < N; n++) {
            for (k = 0; k < M - 1; k++) {
                theta[0][n][k][0] = 0;
                theta[1][n][k][0] = 0;
                theta[2][n][k][0] = 0;
                theta[1][n][k][1] = 0;
                theta[2][n][k][1] = 0;
                theta[0][n][k][2] = 0;
                theta[1][n][k][2] = 0;
            }
        }
        for (n = 0; n < N; n++) {
            for (k = 0; k < M - 1; k++) {
                for (j = 0; j < S; j++) {
                    psi_kl[j][0] = psi(j + 1,(double)k / (double)M);
                    psi_kl[j][1] = psi(j + 1,((double)k + 1) / (double)M);
                    psi_kl[j][2] = psi(j + 1,((double)k + 2) / (double)M);
                    theta[0][n][k][0] += Y[n][j] * psi_kl[j][0];
                    theta[1][n][k][0] += Y[n][j] * psi_kl[j][1];
                    theta[2][n][k][0] += Y[n][j] * psi_kl[j][2];
                    theta[1][n][k][1] += Y[n][j] * psi_kl[j][1];
                    theta[2][n][k][1] += Y[n][j] * psi_kl[j][2];
                    theta[0][n][k][2] += Y[n][j] * psi_kl[j][0];
                    theta[1][n][k][2] += Y[n][j] * psi_kl[j][1];
                }
            }
        }
        for (n = 0; n < N; n++) {
            for (k = 0; k < M - 1; k++) {
                theta[0][n][k][0] += alpha;
                theta[1][n][k][0] += alpha;
                theta[2][n][k][0] += alpha;
                theta[1][n][k][1] += alpha;
                theta[2][n][k][1] += alpha;
                theta[0][n][k][2] += alpha;
                theta[1][n][k][2] += alpha;
            }
        }
        for (n = 0; n < N; n++) {
            for (k = 0; k < M - 1; k++) {
                if (k == 0) {
                    B[n][k][0] = ((double)M * (exp(theta[0][n][k][0]) + 4 * exp(theta[1][n][k][0]) + exp(theta[2][n][k][0]))) / (3);
                    B[n][k][1] = -((double)M * (exp(theta[1][n][k][1]) + exp(theta[2][n][k][1]))) / (2);
                    B[n][k][2] = 0;
                }
                else if (k == M - 2) {
                    B[n][k][0] = ((double)M * (exp(theta[0][n][k][0]) + 4 * exp(theta[1][n][k][0]) + exp(theta[2][n][k][0]))) / (3);
                    B[n][k][1] = 0;
                    B[n][k][2] = -((double)M * (exp(theta[0][n][k][2]) + exp(theta[1][n][k][2]))) / (2);
                }
                else {
                    B[n][k][0] = ((double)M * (exp(theta[0][n][k][0]) + 4 * exp(theta[1][n][k][0]) + exp(theta[2][n][k][0]))) / (3);
                    B[n][k][1] = -((double)M * (exp(theta[1][n][k][1]) + exp(theta[2][n][k][1]))) / (2);;
                    B[n][k][2] = -((double)M * (exp(theta[0][n][k][2]) + exp(theta[1][n][k][2]))) / (2);
                }
            }
        }

        //solve the discretized linear equation
        for (k = 0; k < M - 1; k++) {
            g_hat[k] = 1 / (double)M;
        }
        for (i = 0; i < N; i++) {
            double **B_i;
            B_i = getmemory2(M - 1,3);
            B_i = B[i];
            u_hat[i] = linear_system(M - 1,B_i,g_hat);
        }

        //get solution
        for (i = 0; i < N; i++) {
            u[i] = 0;
            for (k = 1; k < M; k++) {
                if (X >= ((double)k - 1) / (double)M && X <= (double)k / (double)M) {
                    u[i] += u_hat[i][k - 1] * (X - ((double)k - 1) / (double)M) * (double)M;
                }
                else if (X > (double)k / (double)M && X <= ((double)k + 1) / (double)M) {
                    u[i] += u_hat[i][k - 1] * (((double)k + 1) / (double)M - X) * (double)M;
                }
            }
        }
        u_est = 0;
        for (i = 0; i < N; i++) {
            u_est += u[i] / (double)N;
        }

        //stop time measurement
        end_all = clock();
        time_all[m] = (double)(end_all-start_all)/CLOCKS_PER_SEC;
        u_x[m] = u_est;
    }

    //get the mean and variance of the solution and and the average computational time
    if (MAIN > 10) {
        double ave_time = 0;
        double mean = 0;
        double variance = 0;
        for (m = 0; m < MAIN; m++) {
            ave_time += time_all[m] / (double)MAIN;
            mean += u_x[m] / (double)MAIN;
            variance += pow(u_x[m],2) / (double)MAIN;
        }
        variance -= pow(mean,2);
        for (m = 0; m < MAIN; m++) {
            printf("%.8f  ",time_all[m]);
        }
        printf("\n");

        for (m = 0; m < MAIN; m++) {
            printf("%.18f  ",u_x[m]);
        }
        printf("\n");
        printf("%.18f\n",mean);
        printf("%.18f\n",variance);
        printf("%.8f\n",pow(variance,0.5));
        printf("%.8f\n",ave_time);
    }
    else{
        for (m = 0; m < MAIN; m++) {
            printf("%.8f  ",time_all[m]);
        }
        printf("\n");

        for (m = 0; m < MAIN; m++) {
            printf("%.18f  ",u_x[m]);
        }
        printf("\n");

    }


    return 0;
}

//2D memory allocation
double** getmemory2(int n,int m) {
    double** A;
    int i;
    A = malloc(sizeof(double *) * n);
    for (i = 0; i < n; i++) {
        A[i] = malloc(sizeof(double) * m);
    }
    return A;
}

//3D memory allocation
double*** getmemory3(int n,int m,int l) {
    double ***A;
    int i,j;
    A = malloc(sizeof(double **) * n);
    for (i = 0; i < n; i++) {
        A[i] = getmemory2(m,l);
    }
    return A;
}

//4D memory allocation
double**** getmemory4(int o,int n,int m,int l) {
    double ****A;
    int i,j,k;
    A = malloc(sizeof(double ***) * o);
    for (i = 0; i < o; i++) {
        A[i] = getmemory3(n,m,l);
    }
    return A;
}

//psi function
double psi(int j,double X_i) {
    double res;
    res = (1 / pow((double)j,2)) * sin(2 * M_PI * (double)j * X_i);
    return res;
}

//obtain the lower triangular matrix by Cholesky decomposition
double*** LU(int n,double **A){
    int i;
    double ***LU;
    LU = getmemory3(2,n,3);
    for (i = 0; i < n; i++) {
        LU[0][i][0] = 1;
        LU[0][i][1] = 0;
        LU[1][i][2] = 0;
    }
    LU[1][0][0] = A[0][0];
    LU[0][0][2] = 0;
    LU[1][n - 1][1] = 0;
    for (i = 1; i < n; i++) {
        LU[0][i][2] = A[i][2] / LU[1][i - 1][0];
        LU[1][i - 1][1] = A[i - 1][1];
        LU[1][i][0] = A[i][0] - LU[0][i][2] * LU[1][i - 1][1];
    }
    return LU;
}

//solve a linear equation
double* linear_system(int n,double**A,double *b){
    int i,l;
    double ***LU_A;
    double **L_A,**U_A;
    LU_A = getmemory3(2,n,3);
    L_A = getmemory2(n,3);
    U_A = getmemory2(n,3);
    LU_A = LU(n,A);
    L_A = LU_A[0];
    U_A = LU_A[1];

    double *y;
    y = malloc(sizeof(double *) * n);
    y[0] = b[0];
    for (i = 1; i < n; i++) {
        y[i] = (b[i] - L_A[i][2] * y[i - 1]) / L_A[i][0];
    }

    double *x;
    x = malloc(sizeof(double *) * n);
    x[n - 1] = y[n - 1] / U_A[n - 1][0];
    for (i = n -2; i >= 0; i--) {
        x[i] = (y[i] - U_A[i][1] * x[i + 1]) / U_A[i][0];
    }

    return x;
}
