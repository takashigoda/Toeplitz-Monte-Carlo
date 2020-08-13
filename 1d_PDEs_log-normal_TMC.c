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
double** fft(int condition,double **z);
double*** LU(int n,double **A);
double* linear_system(int n,double **A,double *b);

int main(void) {
    //initialization
    int m,i,j,k,l,n,mup;
    int condition = 0;
    double X = 0.5;
    double alpha = 2.0;
    int seed;
    int diff = 0;
    double *time_all,*u_x;
    clock_t start_all,end_all;
    double u_est = 0;
    double *x,*y;
    double *xsub,*ysub;
    double ****theta;
    double **z,**fft_z,**a_t,**fft_a_t,**fft_p,**p,***B;
    double *g_hat,**u_hat;
    double *u;

    time_all = (double *)malloc(sizeof(double) * MAIN);
    u_x = (double *)malloc(sizeof(double) * MAIN);
    x = (double *)malloc(sizeof(double) * N);
    y = (double *)malloc(sizeof(double) * S);
    xsub = (double *)malloc(sizeof(double) * S);
    ysub = (double *)malloc(sizeof(double) * S);
    z = getmemory2(2 * S,2);
    fft_z = getmemory2(2 * S,2);
    a_t = getmemory2(2 * S,2);
    fft_a_t = getmemory2(2 * S,2);
    p = getmemory2(2 * S,2);
    fft_p = getmemory2(2 * S,2);
    theta = getmemory4(3,N,(M -1),3);
    B = getmemory3(N,(M -1),3);
    g_hat = (double *)malloc(sizeof(double) * (M - 1));
    u_hat = getmemory2(N,(M -1));
    u = (double *)malloc(sizeof(double) * N);

    printf("%d\n",N);
    printf("\n");

    for (m = 0; m < MAIN; m++) {
        //set seed of random numbers
        sfmt_t sfmt;
        seed = (m + diff) * 1000;
        sfmt_init_gen_rand(&sfmt, seed);

        //start time measurement
        start_all = clock();

        mup = N / S;

        //generate a random Toeplitz matrix with standard normally distributed samples
        for (n = 0; n < N; n++) {
            if (condition == 0) {
                x[n] = sqrt(-2.0 * log(sfmt_genrand_real2(&sfmt))) * sin(2.0 * M_PI * sfmt_genrand_real2(&sfmt));
            }
            else {
                x[n] = sfmt_genrand_real2(&sfmt) - 0.5;
            }
        }
        for (j = 1; j < S; j++) {
            if (condition == 0) {
                y[j] = sqrt(-2.0 * log(sfmt_genrand_real2(&sfmt))) * sin(2.0 * M_PI * sfmt_genrand_real2(&sfmt));
            }
            else {
                y[j] = sfmt_genrand_real2(&sfmt) - 0.5;
            }
        }
        y[0] = x[0];

        //convert into a circulant matrix
        for (n = 1; n <= mup; n++) {
            for (j = 0; j < S; j++) {
                xsub[j] = x[(n - 1) * S + j];
            }
            if (n == 1) {
                for (j = 0; j < S; j++) {
                    ysub[j] = y[j];
                }
            }
            else{
                for (j = 0; j < S; j++) {
                    ysub[j] = x[(n - 1) * S - j];
                }
            }
            for(j = 0; j < 2 * S; j++) {
                z[j][1] = 0;
                a_t[j][0] = 0;
                a_t[j][1] = 0;
                if (j < S) {
                    z[j][0] = xsub[j];
                }
                else if (j == S){
                    z[j][0] = 0;
                }
                else{
                    z[j][0] = ysub[2 * S - j];
                }
            }

            //compute the stiff matrix B
            //l=k
            for (k = 0; k < (M -1); k++) {
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < S; j++) {
                        a_t[j][0] = psi(j + 1,((double)k + (double)i) / (double)M);
                    }
                    fft_z = fft(0,z);
                    fft_a_t = fft(0,a_t);
                    for (j = 0; j < 2 * S; j++) {
                        fft_p[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                        fft_p[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                    }
                    p = fft(1,fft_p);
                    for (j = 0; j < S; j++) {
                        theta[i][(n - 1) * S + j][k][0] = C * p[j][0] + alpha;
                    }
                }
                //quadrature
                for (j = 0; j < S; j++) {
                    B[(n - 1) * S + j][k][0] = ((double)M * (exp(theta[0][(n - 1) * S + j][k][0]) + 4 * exp(theta[1][(n - 1) * S + j][k][0]) + exp(theta[2][(n - 1) * S + j][k][0]))) / (3);
                }
            }

            //l=k+1
            for (k = 0; k < M - 2; k++) {
                for (i = 1; i < 3; i++) {
                    for (j = 0; j < S; j++) {
                        a_t[j][0] = psi(j + 1,((double)k + (double)i) / (double)M);
                    }
                    fft_z = fft(0,z);
                    fft_a_t = fft(0,a_t);
                    for (j = 0; j < 2 * S; j++) {
                        fft_p[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                        fft_p[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                    }
                    p = fft(1,fft_p);
                    for (j = 0; j < S; j++) {
                        theta[i][(n - 1) * S + j][k][1] =  C * p[j][0] + alpha;
                    }
                }
                //quadrature
                for (j = 0; j < S; j++) {
                    B[(n - 1) * S + j][k][1] = -((double)M * (exp(theta[1][(n - 1) * S + j][k][1]) + exp(theta[2][(n - 1) * S + j][k][1]))) / (2);
                }
            }

            //l=k-1
            for (k = 1; k < (M -1); k++) {
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < S; j++) {
                        a_t[j][0] = psi(j + 1,((double)k + (double)i) / (double)M);
                    }
                    fft_z = fft(0,z);
                    fft_a_t = fft(0,a_t);
                    for (j = 0; j < 2 * S; j++) {
                        fft_p[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                        fft_p[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                    }
                    p = fft(1,fft_p);
                    for (j = 0; j < S; j++) {
                        theta[i][(n - 1) * S + j][k][2] =  C * p[j][0] + alpha;
                    }
                }
                //quadrature
                for (j = 0; j < S; j++) {
                    B[(n - 1) * S + j][k][2] = -((double)M * (exp(theta[0][(n - 1) * S + j][k][2]) + exp(theta[1][(n - 1) * S + j][k][2]))) / (2);
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
                else if (X >= (double)k / (double)M && X <= ((double)k + 1) / (double)M) {
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
            printf("%f  ",time_all[m]);
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
            printf("%f  ",time_all[m]);
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
    A = (double **)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++) {
        A[i] = (double *)malloc(sizeof(double) * m);
    }
    return A;
}

//3D memory allocation
double*** getmemory3(int n,int m,int l) {
    double ***A;
    int i,j;
    A = (double ***)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++) {
        A[i] = getmemory2(m,l);
    }
    return A;
}

//4D memory allocation
double**** getmemory4(int o,int n,int m,int l) {
    double ****A;
    int i,j,k;
    A = (double ****)malloc(sizeof(double) * o);
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

//function on FFT
double** fft(int condition,double **z){
    int i;
    double **result;
    fftw_complex *in, *out;
    fftw_plan p;

    result = getmemory2(2 * S,2);

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * S));
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (2 * S));
    if (condition == 0) {
        p = fftw_plan_dft_1d(2 * S, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    }
    else if (condition == 1){
        p = fftw_plan_dft_1d(2 * S, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    /* create sin wave of 440Hz */
    for(i = 0;i < 2 * S;i++){
        in[i][0] = z[i][0];
        in[i][1] = z[i][1];
    }

    fftw_execute(p); /* repeat as needed */

    fftw_destroy_plan(p);
    fftw_free(in);

    if (condition == 0) {
        for (i = 0 ; i < 2 * S ; i ++){
            result[i][0] = out[i][0];
            result[i][1] = out[i][1];
        }
    }
    else if(condition == 1){
        for (i = 0; i < 2 * S; i++) {
            result[i][0] = out[i][0] / (2 * S);
            result[i][1] = out[i][1] / (2 * S);
        }
    }


    fftw_free(out);
    return result;
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
    int i;
    double ***LU_A;
    double **L_A,**U_A;
    int l;
    LU_A = getmemory3(2,n,3);
    L_A = getmemory2(n,3);
    U_A = getmemory2(n,3);
    LU_A = LU(n,A);
    L_A = LU_A[0];
    U_A = LU_A[1];

    double *y;
    y = (double *)malloc(sizeof(double) * n);
    y[0] = b[0];
    for (i = 1; i < n; i++) {
        y[i] = (b[i] - L_A[i][2] * y[i - 1]) / L_A[i][0];
    }

    double *x;
    x = (double *)malloc(sizeof(double) * n);
    x[n - 1] = y[n - 1] / U_A[n - 1][0];
    for (i = n -2; i >= 0; i--) {
        x[i] = (y[i] - U_A[i][1] * x[i + 1]) / U_A[i][0];
    }

    return x;
}
