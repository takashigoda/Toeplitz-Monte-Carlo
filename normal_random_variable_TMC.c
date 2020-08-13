#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "SFMT.h"
#include "fftw3.h"

// N: sample size, S: dimension
// MAIN: number of independent runs
#define N (1024)
#define S (1024)
#define MAIN (10)

double** getmemory(int n,int m);
double** fft(int condition,double **z);

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
    int mup;
    double *x,*y;
    double *xsub,*ysub;
    double **B,**z,**fft_z,**a_t,**fft_a_t,**fft_p,**p;

    time_all = malloc(sizeof(double *) * MAIN);
    H = getmemory(S,S);
    tH = getmemory(S,S);
    C = getmemory(S,S);
    A = getmemory(S,S);
    x = malloc(sizeof(double *) * N);
    y = malloc(sizeof(double *) * S);
    xsub = malloc(sizeof(double *) * S);
    ysub = malloc(sizeof(double *) * S);
    z = getmemory(2 * S,2);
    fft_z = getmemory(2 * S,2);
    a_t = getmemory(2 * S,2);
    fft_a_t = getmemory(2 * S,2);
    p = getmemory(2 * S,2);
    fft_p = getmemory(2 * S,2);
    B = getmemory(N,S);

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

    free(H);
    free(tH);
    free(C);

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

        mup = N/S;
        //generate a random Toeplitz matrix with standard normally distibuted samples
        for (i = 0; i < N; i++) {
            x[i] = sqrt(-2.0 * log(sfmt_genrand_real2(&sfmt))) * sin(2.0 * M_PI * sfmt_genrand_real2(&sfmt));
        }
        for (i = 1; i < S; i++) {
            y[i] = sqrt(-2.0 * log(sfmt_genrand_real2(&sfmt))) * sin(2.0 * M_PI * sfmt_genrand_real2(&sfmt));
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

            //generate multivariate normal vectors (fast matrix-vector multiplication with FFT)
            for (j = 0; j < S; j++) {
                for (k = 0; k < S; k++) {
                    a_t[k][0] = A[k][j];
                }
                fft_z = fft(0,z);
                fft_a_t = fft(0,a_t);
                for (k = 0; k < 2 * S; k++) {
                    fft_p[k][0] = fft_z[k][0] * fft_a_t[k][0] - fft_z[k][1] * fft_a_t[k][1];
                    fft_p[k][1] = fft_z[k][0] * fft_a_t[k][1] + fft_z[k][1] * fft_a_t[k][0];
                }
                p = fft(1,fft_p);
                for (k = 0; k < S; k++) {
                    B[(i - 1) * S + k][j] = p[k][0];
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

//function on FFT
double** fft(int condition,double **z){
    int i;
    double **result;
    fftw_complex *in, *out;
    fftw_plan p;

    result = getmemory(2 * S,2);

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
