#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "SFMT.h"
#include "fftw3.h"
#include "nrutil.h"
#include <unistd.h>

// N: sample size, S: dimension, M: number of finite elements
// MAIN: number of independent runs
#define N (64)
#define S (64)
#define M (8)
#define MAIN (25)

double** getmemory2(int n,int m);
double*** getmemory3(int n,int m,int l);
int** int_getmemory2(int n,int m);
double** fft(int condition,double **z);
double integral_of_sin(int k,double b,double a);
double integral_of_sin_cos(int k1,int k2,double b,double a);
double integral_of_sin_sin(int k1,int k2,double b,double a);
void selectionSort(int *nodes,int *indices, int array_size);
double*** LU(int n,double **A);
double* linear_system(int n,double **A,double *b);
double p_q_0(int k1,int k2,int p,int q);
double p_q_1(int k1,int k2,int p,int q);
double p_q_2(int k1,int k2,int p,int q);
double p_q_3(int k1,int k2,int p,int q);
double p_q_4(int k1,int k2,int p,int q);
void sprsin(double **a,double *sa,int *ija);
void linbcg(double *b,double *x,int itol,double tol,int itmax,int *iter,double *err,double *sa,int *ija);
void atimes(int n,double *x,double *r,int itrnsp,double *sa,int *ija);
double snrm(int n,double *sx,int itol);
void dsprsax(double *sa,int *ija,double *x,double *b,int n);
void dsprstx(double *sa,int *ija,double *x,double *b,int n);

int main(void) {
    //initialization
    double x1 = 0.5;
    double x2 = 0.5;
    int m,i,j,k,l,p,q,mup;
    int seed;
    int diff = 0;
    double *time_all,*u_x,*err_all,*n_err;
    int itol = 1;
    double tol = 1.0e-5;
    int itmax = 10000;
    int *iter;
    double *err;
    clock_t start_all,end_all;
    double u_est = 0;
    double ***Amat;
    double *x,*y;
    double *xsub,*ysub;
    double ***B,**z,**fft_z,**a_t,**fft_a_t,**fft_r,**r;
    double *sb;
    int *ijb;
    int nmax = 5 * M * M - 14 * M + 10;
    double *g_hat,**u_hat;
    double *u;

    time_all = malloc(sizeof(double *) * MAIN);
    err_all = malloc(sizeof(double *) * MAIN);
    n_err = malloc(sizeof(double *) * N);
    u_x = malloc(sizeof(double *) * MAIN);
    Amat = getmemory3(S + 1,(M - 1) * (M - 1),5);
    x = malloc(sizeof(double *) * N);
    y = malloc(sizeof(double *) * S);
    xsub = malloc(sizeof(double *) * S);
    ysub = malloc(sizeof(double *) * S);
    z = getmemory2(2 * S,2);
    fft_z = getmemory2(2 * S,2);
    a_t = getmemory2(2 * S,2);
    fft_a_t = getmemory2(2 * S,2);
    r = getmemory2(2 * S,2);
    fft_r = getmemory2(2 * S,2);
    sb = malloc(sizeof(double *) * nmax);
    ijb = malloc(sizeof(int *) * nmax);
    B = getmemory3(N,(M - 1) * (M - 1),5);
    g_hat = malloc(sizeof(double *) * (M - 1) * (M - 1));
    u_hat = getmemory2(N,(M - 1) * (M - 1));
    u = malloc(sizeof(double *) * N);

    //lambda_j
    int virtual_mesh_size;
    int *sort_indices;
    int **node;
    int *sort_node;
    int *lambda;
    int **k_node;

    virtual_mesh_size = (int)(pow(2 * (double)S,0.5) + 0.9999999999);
    sort_indices = malloc(sizeof(int *) * (int)pow((virtual_mesh_size),2));
    node = int_getmemory2(2,(int)pow(virtual_mesh_size,2));
    sort_node = malloc(sizeof(int *) * (int)pow((virtual_mesh_size),2));
    lambda = malloc(sizeof(int *) * S);
    k_node = int_getmemory2(2,S);
    for (i = 0; i < (int)pow(virtual_mesh_size,2); i++) {
        sort_indices[i] = i;
    }
    for (j = 0; j < virtual_mesh_size; j++) {
        for (k = 0; k < virtual_mesh_size; k++) {
            node[0][j * virtual_mesh_size + k] = j + 1;
            node[1][j * virtual_mesh_size + k] = k + 1;
        }
    }
    for (i = 0; i < (int)pow(virtual_mesh_size,2); i++) {
        sort_node[i] = pow(node[0][i],2) + pow(node[1][i],2);
    }
    selectionSort(sort_node, sort_indices, (int)pow(virtual_mesh_size,2));
    for (i = 0; i < S; i++) {
        lambda[i] = sort_node[i];
        k_node[0][i] = node[0][sort_indices[i]];
        k_node[1][i] = node[1][sort_indices[i]];
    }

    //compute the symmetric matrix A
    for (p = 0; p < M - 1; p++) {
        for (q = 0; q < M - 1; q++) {
            if (p == 0) {
                if (q == 0) {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = 0;
                    Amat[0][q * (M - 1) + p][2] = 0;
                    Amat[0][q * (M - 1) + p][3] = -1;
                    Amat[0][q * (M - 1) + p][4] = -1;
                }
                else if (q == M -2) {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = -1;
                    Amat[0][q * (M - 1) + p][2] = 0;
                    Amat[0][q * (M - 1) + p][3] = -1;
                    Amat[0][q * (M - 1) + p][4] = 0;
                }
                else {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = -1;
                    Amat[0][q * (M - 1) + p][2] = 0;
                    Amat[0][q * (M - 1) + p][3] = -1;
                    Amat[0][q * (M - 1) + p][4] = -1;
                }
            }
            else if (p == M - 2) {
                if (q == 0) {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = 0;
                    Amat[0][q * (M - 1) + p][2] = -1;
                    Amat[0][q * (M - 1) + p][3] = 0;
                    Amat[0][q * (M - 1) + p][4] = -1;
                }
                else if (q == M -2) {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = -1;
                    Amat[0][q * (M - 1) + p][2] = -1;
                    Amat[0][q * (M - 1) + p][3] = 0;
                    Amat[0][q * (M - 1) + p][4] = 0;
                }
                else {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = -1;
                    Amat[0][q * (M - 1) + p][2] = -1;
                    Amat[0][q * (M - 1) + p][3] = 0;
                    Amat[0][q * (M - 1) + p][4] = -1;
                }
            }
            else {
                if (q == 0) {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = 0;
                    Amat[0][q * (M - 1) + p][2] = -1;
                    Amat[0][q * (M - 1) + p][3] = -1;
                    Amat[0][q * (M - 1) + p][4] = -1;
                }
                else if (q == M -2) {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = -1;
                    Amat[0][q * (M - 1) + p][2] = -1;
                    Amat[0][q * (M - 1) + p][3] = -1;
                    Amat[0][q * (M - 1) + p][4] = 0;
                }
                else {
                    Amat[0][q * (M - 1) + p][0] = 4;
                    Amat[0][q * (M - 1) + p][1] = -1;
                    Amat[0][q * (M - 1) + p][2] = -1;
                    Amat[0][q * (M - 1) + p][3] = -1;
                    Amat[0][q * (M - 1) + p][4] = -1;
                }
            }
        }
    }
    for (j = 1; j <= S; j++) {
        for (p = 0; p < M - 1; p++) {
            for (q = 0; q < M - 1; q++) {
                if (p == 0) {
                    if (q == 0) {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = 0;
                        Amat[j][q * (M - 1) + p][2] = 0;
                        Amat[j][q * (M - 1) + p][3] = p_q_3(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][4] = p_q_4(k_node[0][j - 1],k_node[1][j - 1],p,q);
                    }
                    else if (q == M -2) {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = p_q_1(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][2] = 0;
                        Amat[j][q * (M - 1) + p][3] = p_q_3(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][4] = 0;
                    }
                    else {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = p_q_1(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[0][q * (M - 1) + p][2] = 0;
                        Amat[j][q * (M - 1) + p][3] = p_q_3(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][4] = p_q_4(k_node[0][j - 1],k_node[1][j - 1],p,q);
                    }
                }
                else if (p == M - 2) {
                    if (q == 0) {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = 0;
                        Amat[j][q * (M - 1) + p][2] = p_q_2(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][3] = 0;
                        Amat[j][q * (M - 1) + p][4] = p_q_4(k_node[0][j - 1],k_node[1][j - 1],p,q);
                    }
                    else if (q == M -2) {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = p_q_1(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][2] = p_q_2(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][3] = 0;
                        Amat[j][q * (M - 1) + p][4] = 0;
                    }
                    else {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = p_q_1(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][2] = p_q_2(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][3] = 0;
                        Amat[j][q * (M - 1) + p][4] = p_q_4(k_node[0][j - 1],k_node[1][j - 1],p,q);
                    }
                }
                else {
                    if (q == 0) {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = 0;
                        Amat[j][q * (M - 1) + p][2] = p_q_2(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][3] = p_q_3(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][4] = p_q_4(k_node[0][j - 1],k_node[1][j - 1],p,q);
                    }
                    else if (q == M -2) {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = p_q_1(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][2] = p_q_2(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][3] = p_q_3(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][4] = 0;
                    }
                    else {
                        Amat[j][q * (M - 1) + p][0] = p_q_0(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][1] = p_q_1(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][2] = p_q_2(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][3] = p_q_3(k_node[0][j - 1],k_node[1][j - 1],p,q);
                        Amat[j][q * (M - 1) + p][4] = p_q_4(k_node[0][j - 1],k_node[1][j - 1],p,q);
                    }
                }
            }
        }
    }

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

        //generate a random Toeplitz matrix with uniformly distributed samples
        for (i = 0; i < N; i++) {
            x[i] = sfmt_genrand_real2(&sfmt) - 0.5;
        }
        for (i = 0; i < S; i++) {
            y[i] = sfmt_genrand_real2(&sfmt) - 0.5;
        }
        y[0] = x[0];

        //convert into a circulant matrix
        for (i = 1; i <= mup; i++) {
            for (j = 0; j < S; j++) {
                xsub[j] = x[(i - 1) * S + j];
            }
            if (i == 1) {
                for (j = 0; j < S; j++) {
                    ysub[j] = y[j];
                }
            }
            else{
                for (j = 0; j < S; j++) {
                    ysub[j] = x[(i - 1) * S - j];
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
            //(p,q) and (p,q)
            for (k = 0; k < (M - 1) * (M - 1); k++) {
                for (j = 0; j < S; j++) {
                    a_t[j][0] = Amat[j + 1][k][0];
                }
                fft_z = fft(0,z);
                fft_a_t = fft(0,a_t);
                for (j = 0; j < 2 * S; j++) {
                    fft_r[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                    fft_r[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                }
                r = fft(1,fft_r);
                for (j = 0; j < S; j++) {
                    B[(i - 1) * S + j][k][0] = Amat[0][k][0] + r[j][0];
                }
            }

            //(p,q) and (p,q-1)
            for (k = 0; k < (M - 1) * (M - 1); k++) {
                for (j = 0; j < S; j++) {
                    a_t[j][0] = Amat[j + 1][k][1];
                }
                fft_z = fft(0,z);
                fft_a_t = fft(0,a_t);
                for (j = 0; j < 2 * S; j++) {
                    fft_r[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                    fft_r[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                }
                r = fft(1,fft_r);
                for (j = 0; j < S; j++) {
                    B[(i - 1) * S + j][k][1] = Amat[0][k][1] + r[j][0];
                }
            }

            //(p,q) and (p-1,q)
            for (k = 0; k < (M - 1) * (M - 1); k++) {
                for (j = 0; j < S; j++) {
                    a_t[j][0] = Amat[j + 1][k][2];
                }
                fft_z = fft(0,z);
                fft_a_t = fft(0,a_t);
                for (j = 0; j < 2 * S; j++) {
                    fft_r[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                    fft_r[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                }
                r = fft(1,fft_r);
                for (j = 0; j < S; j++) {
                    B[(i - 1) * S + j][k][2] = Amat[0][k][2] + r[j][0];
                }
            }

            //(p,q) and (p+1,q)
            for (k = 0; k < (M - 1) * (M - 1); k++) {
                for (j = 0; j < S; j++) {
                    a_t[j][0] = Amat[j + 1][k][3];
                }
                fft_z = fft(0,z);
                fft_a_t = fft(0,a_t);
                for (j = 0; j < 2 * S; j++) {
                    fft_r[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                    fft_r[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                }
                r = fft(1,fft_r);
                for (j = 0; j < S; j++) {
                    B[(i - 1) * S + j][k][3] = Amat[0][k][3] + r[j][0];
                }
            }

            //(p,q) and (p,q+1)
            for (k = 0; k < (M - 1) * (M - 1); k++) {
                for (j = 0; j < S; j++) {
                    a_t[j][0] = Amat[j + 1][k][4];
                }
                fft_z = fft(0,z);
                fft_a_t = fft(0,a_t);
                for (j = 0; j < 2 * S; j++) {
                    fft_r[j][0] = fft_z[j][0] * fft_a_t[j][0] - fft_z[j][1] * fft_a_t[j][1];
                    fft_r[j][1] = fft_z[j][0] * fft_a_t[j][1] + fft_z[j][1] * fft_a_t[j][0];
                }
                r = fft(1,fft_r);
                for (j = 0; j < S; j++) {
                    B[(i - 1) * S + j][k][4] = Amat[0][k][4] + r[j][0];
                }
            }
        }

        //solve the discretized linear equation
        for (p = 0; p < (M - 1); p++) {
            for (q = 0; q < M - 1; q++) {
                g_hat[q * (M - 1) + p] = (100 * (double)p + 100) / ((double)M * (double)M * (double)M);
            }
        }

        for (i = 0; i < N; i++) {
            double **B_i;
            B_i = getmemory2((M - 1) * (M - 1),5);
            B_i = B[i];
            sprsin(B_i,sb,ijb);
            for (j = 0; j < (M - 1) * (M - 1); j++) {
                u_hat[i][j] = 0;
            }
            //BiCGSTAB
            linbcg(g_hat,u_hat[i],itol,tol,itmax,iter,err,sb,ijb);
            if (m == 0) {
              if (i == 0) {
                printf("%d\n",*iter);
              }
            }
            n_err[i] = *err;
        }

        //get solution
        for (i = 0; i < N; i++) {
            u[i] = 0;
            for (q = 0; q < M - 1; q++) {
                for (p = 0; p < M - 1; p++) {
                    if (x1 >= (double)p / (double)M && x1 <= ((double)p + 1) / (double)M && x2 >= (double)q / (double)M && x2 <= x1 + ((double)q - (double)p) / (double)M) {
                        u[i] += u_hat[i][q * (M - 1) + p] * ((double)M * x2 - q);
                    }
                    else if (x1 >= (double)p / (double)M && x1 <= ((double)p + 1) / (double)M && x2 <= ((double)q + 1) / (double)M && x2 >= x1 + ((double)q - (double)p) / (double)M) {
                        u[i] += u_hat[i][q * (M - 1) + p] * ((double)M * x1 - p);
                    }
                    else if (x1 >= (double)p / (double)M && x1 <= ((double)p + 1) / (double)M && x2 >= ((double)q + 1) / (double)M && x2 <= x1 + ((double)q - (double)p + 1) / (double)M) {
                        u[i] += u_hat[i][q * (M - 1) + p] * ((double)M * x1 - (double)M * x2 + 1 + q - p);
                    }
                    else if (x1 >= ((double)p + 1) / (double)M && x1 <= ((double)p + 2) / (double)M && x2 <= ((double)q + 1) / (double)M && x2 >= x1 + ((double)q - (double)p - 1) / (double)M) {
                        u[i] += u_hat[i][q * (M - 1) + p] * (-(double)M * x1 + (double)M * x2 + 1 + p - q);
                    }
                    else if (x1 >= ((double)p + 1) / (double)M && x1 <= ((double)p + 2) / (double)M && x2 >= ((double)q + 1) / (double)M && x2 <= x1 + ((double)q - (double)p) / (double)M) {
                        u[i] += u_hat[i][q * (M - 1) + p] * (-(double)M * x1 + 2 + p);
                    }
                    else if (x1 >= ((double)p + 1) / (double)M && x1 <= ((double)p + 2) / (double)M && x2 <= ((double)q + 2) / (double)M && x2 >= x1 + ((double)q - (double)p) / (double)M) {
                        u[i] += u_hat[i][q * (M - 1) + p] * (-(double)M * x2 + q + 2);
                    }
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
        err_all[m] = 0;
        for (i = 0; i < N; i++) {
            err_all[m] += n_err[i] / (double)N;
        }

    }

    //get the mean and variance of the solution and and the average computational time
    double ave_err = 0;
    double ave_time = 0;
    double mean = 0;
    double variance = 0;
    if (MAIN > 10) {
        for (m = 0; m < MAIN; m++) {
            ave_err += err_all[m] / (double)MAIN;
            ave_time += time_all[m] / (double)MAIN;
            mean += u_x[m] / (double)MAIN;
            variance += pow(u_x[m],2) / (double)MAIN;
        }
        variance -= pow(mean,2);
        printf("%.8f\n",mean);
        printf("%.12f\n",variance);
        printf("%.12f\n",pow(variance,0.5));
        printf("%.8f\n",ave_time);
        printf("err:%.20f\n",ave_err);
    }
    else{
        for (m = 0; m < MAIN; m++) {
            printf("%f  ",time_all[m]);
        }
        printf("\n");
        for (m = 0; m < MAIN; m++) {
            printf("err:%.20f  ",err_all[m]);
        }
        printf("\n");
        for (m = 0; m < MAIN; m++) {
            printf("%f  ",u_x[m]);
        }
        printf("\n");
        for (m = 0; m < MAIN; m++) {
            ave_time += time_all[m] / (double)MAIN;
            mean += u_x[m] / (double)MAIN;
            variance += pow(u_x[m],2) / (double)MAIN;
        }
        variance -= pow(mean,2);
        printf("%.8f\n",mean);
        printf("%.12f\n",variance);
        printf("%.8f\n",ave_time);
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

//2D memory allocation for integers
int** int_getmemory2(int n,int m) {
    int** A;
    int i;
    A = malloc(sizeof(int *) * n);
    for (i = 0; i < n; i++) {
        A[i] = malloc(sizeof(int) * m);
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

//integral
double integral_of_sin(int k,double b,double a){
    double answer0;
    answer0 = (1 / ((double)k * M_PI)) * (cos(a * (double)k * M_PI) - cos(b * (double)k * M_PI));
    //printf("s:%f\n",answer0);
    return answer0;
}

double integral_of_sin_cos(int k1,int k2,double b,double a){
    double answer1;
    if (k1 == k2) {
        answer1 = (1 / (2 * ((double)k1 + (double)k2) * M_PI)) * (cos(a * ((double)k1 + (double)k2) * M_PI) - cos(b * ((double)k1 + (double)k2) * M_PI));
    }
    else {
        answer1 = (1 / (2 * ((double)k1 + (double)k2) * M_PI)) * (cos(a * ((double)k1 + (double)k2) * M_PI) - cos(b * ((double)k1 + (double)k2) * M_PI)) + (1 / (2 * ((double)k1 - (double)k2) * M_PI)) * (cos(a * ((double)k1 - (double)k2) * M_PI) - cos(b * ((double)k1 - (double)k2) * M_PI));;
    }
    //printf("sc:%f\n",answer1);
    return answer1;
}

double integral_of_sin_sin(int k1,int k2,double b,double a){
    double answer2;
    if (k1 == k2) {
        answer2 = 0.5 * (b - a - (1 / (((double)k1 + (double)k2) * M_PI)) * (sin(b * ((double)k1 + (double)k2) * M_PI) - sin(a * ((double)k1 + (double)k2) * M_PI)));
    }
    else {
        answer2 = 0.5 * ((1 / (((double)k1 - (double)k2) * M_PI)) * (sin(b * ((double)k1 - (double)k2) * M_PI) - sin(a * ((double)k1 - (double)k2) * M_PI)) - (1 / (((double)k1 + (double)k2) * M_PI)) * (sin(b * ((double)k1 + (double)k2) * M_PI) - sin(a * ((double)k1 + (double)k2) * M_PI)));
    }
    //printf("ss:%f\n",answer2);
    return answer2;
}

void selectionSort(int *nodes,int *indices, int array_size) {
    int i;
    int j;
    int min;
    int temp1,temp2;

    for (i = 0; i < array_size-1; i++) {
        min = i;
        for (j = i+1; j < array_size; j++) {
            if (nodes[j] < nodes[min]) {
                min = j;
            }
        }

        temp1 = nodes[i];
        nodes[i] = nodes[min];
        nodes[min] = temp1;

        temp2 = indices[i];
        indices[i] = indices[min];
        indices[min] = temp2;
    }
}

//Amat
double p_q_0(int k1,int k2,int p,int q) {
    double answer3;
    answer3 = ((double)M * (double)M / (((double)k1 * (double)k1 + (double)k2 * (double)k2) * ((double)k1 * (double)k1 + (double)k2 * (double)k2) * (double)k2 * M_PI)) * ((cos((double)q * (double)k2 * M_PI / (double)M) + cos(((double)q + 1) * (double)k2 * M_PI / (double)M)) * integral_of_sin(k1,((double)p + 1) / (double)M,(double)p / (double)M) - (cos(((double)q + 1) * (double)k2 * M_PI / (double)M) + cos(((double)q + 2) * (double)k2 * M_PI / (double)M)) * integral_of_sin(k1,((double)p + 2) / (double)M,((double)p + 1) / (double)M) - 2 * cos(((double)q + 1 - (double)p) * (double)k2 * M_PI / (double)M) * integral_of_sin_cos(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M) + 2 * sin(((double)q + 1 - (double)p) * (double)k2 * M_PI / (double)M) * integral_of_sin_sin(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M) + 2 * cos(((double)q - 1 - (double)p) * (double)k2 * M_PI / (double)M) * integral_of_sin_cos(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M) - 2 * sin(((double)q - 1 - (double)p) * (double)k2 * M_PI / (double)M) * integral_of_sin_sin(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M));
    return answer3;
}

double p_q_1(int k1,int k2,int p,int q) {
    double answer4;
    answer4 = -(double)M * (double)M / (((double)k1 * (double)k1 + (double)k2 * (double)k2) * ((double)k1 * (double)k1 + (double)k2 * (double)k2) * (double)k2 * M_PI) * (cos((double)q * k2 * M_PI / (double)M) * integral_of_sin(k1,((double)p + 1) / (double)M,(double)p / (double)M) - cos(((double)q + 1) * k2 * M_PI / (double)M) * integral_of_sin(k1,((double)p + 2) / (double)M,((double)p + 1) / (double)M) - cos(((double)q - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_cos(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M) + sin(((double)q - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_sin(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M) + cos(((double)q - 1 - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_cos(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M) - sin(((double)q - 1 - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_sin(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M));
    return answer4;
}

double p_q_2(int k1,int k2,int p,int q) {
    double answer5;
    answer5 = -(double)M * (double)M / (((double)k1 * (double)k1 + (double)k2 * (double)k2) * ((double)k1 * (double)k1 + (double)k2 * (double)k2) * (double)k2 * M_PI) * ((cos(((double)q - (double)p) * k2 * M_PI / (double)M) - cos(((double)q + 1 - (double)p) * k2 * M_PI / (double)M)) * integral_of_sin_cos(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M) - (sin(((double)q - (double)p) * k2 * M_PI / (double)M) - sin(((double)q + 1 - (double)p) * k2 * M_PI / (double)M)) * integral_of_sin_sin(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M));
    return answer5;
}

double p_q_3(int k1,int k2,int p,int q) {
    double answer6;
    answer6 = -(double)M * (double)M / (((double)k1 * (double)k1 + (double)k2 * (double)k2) * ((double)k1 * (double)k1 + (double)k2 * (double)k2) * (double)k2 * M_PI) * ((cos(((double)q - 1 - (double)p) * k2 * M_PI / (double)M) - cos(((double)q - (double)p) * k2 * M_PI / (double)M)) * integral_of_sin_cos(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M) - (sin(((double)q - 1 - (double)p) * k2 * M_PI / (double)M) - sin(((double)q - (double)p) * k2 * M_PI / (double)M)) * integral_of_sin_sin(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M));
    return answer6;
}

double p_q_4(int k1,int k2,int p,int q) {
    double answer7;
    answer7 = -(double)M * (double)M / (((double)k1 * (double)k1 + (double)k2 * (double)k2) * ((double)k1 * (double)k1 + (double)k2 * (double)k2) * (double)k2 * M_PI) * (cos(((double)q + 1) * k2 * M_PI / (double)M) * integral_of_sin(k1,((double)p + 1) / (double)M,(double)p / (double)M) - cos(((double)q + 2) * k2 * M_PI / (double)M) * integral_of_sin(k1,((double)p + 2) / (double)M,((double)p + 1) / (double)M) - cos(((double)q + 1 - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_cos(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M) + sin(((double)q + 1 - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_sin(k1,k2,((double)p + 1) / (double)M,(double)p / (double)M) + cos(((double)q - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_cos(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M) - sin(((double)q - (double)p) * k2 * M_PI / (double)M) * integral_of_sin_sin(k1,k2,((double)p + 2) / (double)M,((double)p + 1) / (double)M));
    return answer7;
}

void sprsin(double **a,double *sa,int *ija) {
    int ii,jj,kk,nn,pp,qq,rr,ss;
    nn = (M - 1) * (M - 1);
    for (jj = 0; jj < nn; jj++) {
        sa[jj] = a[jj][0];
    }
    rr = nn;
    sa[nn] = 0;
    for (qq = 0; qq < M - 1; qq++) {
        for (pp = 0; pp < M - 1; pp++) {
            if (qq == 0) {
                if (pp == 0) {
                    for (ii = 0; ii < 2; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][3 + ii];
                    }
                }
                else if (pp == M - 2) {
                    for (ii = 0; ii < 2; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][2 + 2 * ii];
                    }
                }
                else {
                    for (ii = 0; ii < 3; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][2 + ii];
                    }
                }
            }
            else if (qq == M - 2) {
                if (pp == 0) {
                    for (ii = 0; ii < 2; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][1 + 2 * ii];
                    }
                }
                else if (pp == M - 2) {
                    for (ii = 0; ii < 2; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][1 + ii];
                    }
                }
                else {
                    for (ii = 0; ii < 3; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][1 + ii];
                    }
                }
            }
            else {
                if (pp == 0) {
                    rr = rr + 1;
                    sa[rr] = a[qq * (M - 1) + pp][1];
                    for (ii = 0; ii < 2; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][3 + ii];
                    }
                }
                else if (pp == M - 2) {
                    for (ii = 0; ii < 2; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][1 + ii];
                    }
                    rr = rr + 1;
                    sa[rr] = a[qq * (M - 1) + pp][4];
                }
                else {
                    for (ii = 0; ii < 4; ii++) {
                        rr = rr + 1;
                        sa[rr] = a[qq * (M - 1) + pp][1 + ii];
                    }
                }
            }
        }
    }

    ss = 0;
    for (qq = 0; qq < M - 1; qq++) {
        for (pp = 0; pp < M - 1; pp++) {
            if (qq == 0) {
                if (pp == 0) {
                    ija[ss] = nn + 1;
                    ss = ss + 1;
                }
                else {
                    ija[ss] = nn + 3 * pp;
                    ss = ss + 1;
                }
            }
            else if (qq == M - 2) {
                if (pp == 0) {
                    ija[ss] = nn + 4 * M * M - 15 * M + 14;
                    ss = ss + 1;
                }
                else {
                    ija[ss] = nn + 4 * M * M - 15 * M + 13 + 3 * pp;
                    ss = ss + 1;
                }
            }
            else {
                if (pp == 0) {
                    ija[ss] = nn + 3 * M -4 + (qq - 1) * (4 * M - 6);
                    ss = ss + 1;
                }
                else {
                    ija[ss] = nn + 3 * M + 4 * pp - 5 + (qq - 1) * (4 * M - 6);
                    ss = ss + 1;
                }
            }
        }
    }
    ija[ss] = nn + 4 * M * M - 12 * M + 9;
    for (qq = 0; qq < M - 1; qq++) {
        for (pp = 0; pp < M - 1; pp++) {
            if (qq == 0) {
                if (pp == 0) {
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp + 1 + (M - 2) * ii;
                    }
                }
                else if (pp == M - 2) {
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp - 1 + M * ii;
                    }
                }
                else {
                    ss = ss + 1;
                    ija[ss] = pp - 1;
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp + 1 + (M - 2) * ii;
                    }
                }
            }
            else if (qq == M - 2) {
                if (pp == 0) {
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp - M + 1 + M * ii;
                    }
                }
                else if (pp == M - 2) {
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp - M + 1 + (M - 2) * ii;
                    }
                }
                else {
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp - M + 1 + (M - 2) * ii;
                    }
                    ss = ss + 1;
                    ija[ss] = qq * (M - 1) + pp + 1;
                }
            }
            else {
                if (pp == 0) {
                    ss = ss + 1;
                    ija[ss] = qq * (M - 1) + pp - M + 1;
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp + 1 + (M - 2) * ii;
                    }
                }
                else if (pp == M - 2) {
                    ss = ss + 1;
                    ija[ss] = qq * (M - 1) + pp - M + 1;
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp - 1 + M * ii;
                    }
                }
                else {
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp - M + 1 + (M - 2) * ii;
                    }
                    for (ii = 0; ii < 2; ii++) {
                        ss = ss + 1;
                        ija[ss] = qq * (M - 1) + pp + 1 + (M - 2) * ii;
                    }
                }
            }
        }
    }

}

//BiCGSTAB
void linbcg(double *b,double *x,int itol,double tol,int itmax,int *iter,double *err,double *sa,int *ija) {
    int j1;
    int n1 = (M - 1) * (M - 1);
    double ak,akden,bk,bkden,bknum,bnrm,dxnrm,xnrm,enrm,wk,wkden,wknum;
    double *d,*e,*ee,*f,*sk,*ssk;
    d = malloc(sizeof(double *) * n1);
    e = malloc(sizeof(double *) * n1);
    ee = malloc(sizeof(double *) * n1);
    f = malloc(sizeof(double *) * n1);
    sk = malloc(sizeof(double *) * n1);
    ssk = malloc(sizeof(double *) * n1);

    //calculate initial residual
    *iter=0;
    atimes(n1,x,e,0,sa,ija);
    for (j1 = 0;j1 < n1;j1++) {
        e[j1] = b[j1] - e[j1];
        ee[j1] = e[j1];
    }
    for (j1 = 0;j1 < n1;j1++) {
        f[j1] = e[j1];
    }
    bnrm = snrm(n1,b,itol);
    while (*iter <= itmax) {
        ++(*iter);
        for (bknum = 0.0,j1 = 0;j1 < n1;j1++){
            bknum += e[j1] * ee[j1];
        }
        atimes(n1,f,d,0,sa,ija);
        for (akden =0.0, j1 = 0; j1 < n1; j1++) {
            akden += ee[j1] * d[j1];
        }
        ak = bknum / akden;
        for (j1 = 0; j1 < n1; j1++) {
            sk[j1] = e[j1] - ak * d[j1];
        }
        atimes(n1,sk,ssk,0,sa,ija);
        for (wknum = 0.0,j1 = 0;j1 < n1;j1++){
            wknum += ssk[j1] * sk[j1];
        }
        for (wkden =0.0, j1 = 0; j1 < n1; j1++) {
            wkden += ssk[j1] * ssk[j1];
        }
        wk = wknum / wkden;
        for (j1 = 0; j1 < n1; j1++) {
            x[j1] += ak * f[j1] + wk * sk[j1];
            e[j1] = sk[j1] - wk * ssk[j1];
        }
        bkden = bknum;
        for (bknum = 0.0,j1 = 0;j1 < n1;j1++){
            bknum += e[j1] * ee[j1];
        }
        bk = (ak * bknum) / (wk * bkden);
        for (j1 = 0; j1 < n1; j1++) {
            f[j1] = e[j1] + bk * (f[j1] - wk * d[j1]);
        }
        *err = snrm(n1,e,itol) / bnrm;
        if (*err <= tol){
            break;
        }
    }
}

//L2-norm
double snrm(int n,double *sx,int itol) {
    int i1,isamax;
    double ans;
    if (itol <= 3) {
        ans = 0.0;
        for (i1 = 0;i1 < n;i1++){
            ans += sx[i1] * sx[i1];
        }
        return sqrt(ans);
    }
    else {
        isamax=1;
        for (i1 = 0;i1 < n;i1++) {
            if (fabs(sx[i1]) > fabs(sx[isamax])){
                isamax = i1;
            }
        }
        return fabs(sx[isamax]);
    }
}

void atimes(int n,double *x,double *r,int itrnsp,double *sa,int *ija) {

    if (itrnsp){
        dsprstx(sa,ija,x,r,n);
    }
    else{
        dsprsax(sa,ija,x,r,n);
    }
}

void dsprsax(double *sa,int *ija,double *x,double *b,int n) {
    int i2,k2;
    for (i2 = 0;i2 < n;i2++) {
        b[i2] = sa[i2] * x[i2];
        for (k2 = ija[i2];k2 < ija[i2+1];k2++) {
            b[i2] += sa[k2] * x[ija[k2]];
        }
    }
}

void dsprstx(double *sa,int *ija,double *x,double *b,int n) {
    int i3,j3,k3;
    for (i3 = 0;i3 < n;i3++) {
        b[i3] = sa[i3] * x[i3];
    }
    for (i3 = 0;i3 < n;i3++) {
        for (k3 = ija[i3];k3 < ija[i3+1];k3++) {
            j3 = ija[k3];
            b[j3] += sa[k3] * x[i3];
        }
    }
}
