clc;
clear all;
close all;

addpath("mirt_dctn")
addpath("mex");

d = 3;

precision = "double";

for N = [32, 64, 128, 258, 512, 1024]

    img = randn(N*ones(d,1));
    if strcmp(precision, "single")
        img = single(img);
    endif

    # Benchmark mirt
    tstart_mirt = cputime;
    dct_mirt = mirt_dctn(img);
    dt_mirt = cputime - tstart_mirt;

    # Benchmark FFTW
    tstart_fftw = cputime;
    dct_fftw = fftw_dct3(img, 'forward');
    dt_fftw = cputime - tstart_fftw;

    # Benchmark CUDA
    tstart_cuda = cputime;
    dct_cuda = cudct(img, 'forward');
    dt_cuda = cputime - tstart_cuda;

    # Report
    fprintf("Dimension: %d - N: %d\n", d, N);
    fprintf("MIRT: %.10f\n", dt_mirt);
    fprintf("FFTW: %.10f\n", dt_fftw);
    fprintf("CUDA: %.10f\n", dt_cuda);
    fprintf("\n");

endfor

