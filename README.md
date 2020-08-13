# Toeplitz-Monte-Carlo-in-C

This page lists the C codes used for numerical experiments in the paper: Josef Dick, Takashi Goda and Hiroya Murata, Toeplitz Monte Carlo, submitted to Statistics and Computing (https://arxiv.org/abs/2003.03915).

To run the programs properly, please download a pseudo-random number generator called SFMT from its supporting page http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/ and a version of FFTW from the page http://www.fftw.org/ to perform the fast Fourier transform.

(Section 3.1: Generating points from multivariate Gaussian) "normal_random_variable_stdMC.c" and "normal_random_variable_TMC.c"

(Section 3.2: 1D differential equation with uniform random coeffcients) "1d_PDEs_stdMC.c" and "1d_PDEs_TMC.c"

(Section 3.3: 1D differential equation in the log-normal case) "1d_PDEs_log-normal_stdMC.c" and "1d_PDEs_log-normal_TMC.c"

(Section 3.4: 2D differential equation with random coeffcients) "2d_PDEs_stdMC.c" and "2d_PDEs_TMC.c"
