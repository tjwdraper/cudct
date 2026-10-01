clc;
clear all;
close all;

addpath("mirt_dctn")
addpath("mex/matlab");

d = 2;

precision = "single";

for N = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    input = randn(N*ones(1,d));
    if strcmp(precision, "single")
        input = single(input);
    end

    % Benchmark mirt
    tstart_mirt = tic;
    dct_mirt = mirt_dctn(input);
    dt_mirt = toc(tstart_mirt);

    % Benchmark FFTW
    tstart_fftw = tic;
    dct_fftw = matlab_fftw(input, 'forward');
    dt_fftw = toc(tstart_fftw);

    % Benchmark CUDA
    tstart_cuda = tic;
    dct_cuda = matlab_cudct(input, 'forward');
    dt_cuda = toc(tstart_cuda);

    % Report
    fprintf("Dimension: %d - N: %d\n", d, N);
    fprintf("MIRT: %.10f\n", dt_mirt);
    fprintf("FFTW: %.10f\n", dt_fftw);
    fprintf("CUDA: %.10f\n", dt_cuda);
    fprintf("\n");

end

