clc;
clear all;
close all;

addpath("mirt_dctn")
addpath("mex/matlab");

d = 3;

precision = 'single';

for N = [32, 64, 128, 256, 512]

    input = randn(N*ones(1,d));
    if strcmp(precision, 'single')
        input = single(input);
    end
    
    % Initialize gpu array
    input_gpu = gpuArray(input);
    
    % Allocate size
    matlab_cudct_gpuarray(size(input)', precision);

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
    
    % Benchmark CUDA - GPU only
    tstart_cuda_gpuarr = tic;
    dct_cuda_gpuarr = matlab_cudct_gpuarray(input_gpu, 'forward');
    dt_cuda_gpuarr = toc(tstart_cuda_gpuarr);
    
    % Close the lib
    matlab_cudct_gpuarray();

    % Report
    fprintf("Dimension: %d - N: %d\n", d, N);
    fprintf("MIRT: %.10f\n", dt_mirt);
    fprintf("FFTW: %.10f\n", dt_fftw);
    fprintf("CUDA: %.10f\n", dt_cuda);
    fprintf("CUDA (gpuArray): %.10f\n", dt_cuda_gpuarr);
    fprintf("\n");

end

