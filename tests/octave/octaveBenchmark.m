clc;
clear all;
close all;

addpath("mirt_dctn")
addpath("mex");

d = 3;

precision = "single";

for N = [32, 64, 128, 256, 512]

    input = randn(N*ones(d,1));
    if strcmp(precision, "single")
        input = single(input);
    endif

    # Benchmark mirt
    tstart_mirt = cputime;
    dct_mirt = mirt_dctn(input);
    dt_mirt = cputime - tstart_mirt;

    # Benchmark FFTW
    tstart_fftw = cputime;
    dct_fftw = octave_fftw(input, 'forward');
    dt_fftw = cputime - tstart_fftw;

    # Benchmark CUDA
    tstart_cuda = cputime;
    dct_cuda = octave_cudct(input, 'forward');
    dt_cuda = cputime - tstart_cuda;

    # Report
    fprintf("Dimension: %d - N: %d\n", d, N);
    fprintf("MIRT: %.10f\n", dt_mirt);
    fprintf("FFTW: %.10f\n", dt_fftw);
    fprintf("CUDA: %.10f\n", dt_cuda);
    fprintf("\n");

endfor

