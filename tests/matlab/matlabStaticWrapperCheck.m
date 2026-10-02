clc;
clear all;
close all;

addpath('mirt_dctn')
addpath('mex/matlab');

d = 3;

precision = 'double'; % Either double or single

for N = [32, 64, 128, 256]

    % Create random array of dimension d
    input = randn(N*ones(1,d));
    if strcmp(precision, 'single')
        input = single(input);
    end
    
    % Move to the GPU
    input_gpu = gpuArray(input);

    % Initialize static implementation by passing expect input dimensions of size (Nx1) and precision
    matlab_cudct_static(size(input)', precision);
    matlab_cudct_gpuarray(size(input)', precision);

    % Execute 100 DCT transforms
    dt_cudct = zeros(100, 1);
    dt_cudct_static = zeros(100, 1);
    dt_cudct_static_gpu = zeros(100, 1);
    
    dt_mirt = zeros(100, 1);
    dt_mirt_gpu = zeros(100, 1);
    
    dt_fftw = zeros(100, 1);
    

    for t = 1:100
        % CUDA-DCT implementation incl host-device memory transfer:
        tstart = tic;
        output = matlab_cudct(input, 'forward');
        dt_cudct(t) = toc(tstart);
    end
       
    for t = 1:100
        % CUDA-DCT implementation incl host-device memory transfer, but
        % cudct object already initialized
        tstart = tic;
        output = matlab_cudct_static(input, 'forward');
        dt_cudct_static(t) = toc(tstart);
    end
        
    for t = 1:100
        % CUDA-DCT implementation without host-device memory transfer, and 
        % cudct object already initialized
        tstart = tic;
        output_gpu = matlab_cudct_gpuarray(input_gpu, 'forward');
        dt_cudct_static_gpu(t) = toc(tstart);
    end
        
    for t = 1:100
        % Compare to MIRT
        tstart = tic;
        output = mirt_dctn(input);
        dt_mirt(t) = toc(tstart);
    end
        
    for t = 1:100
        % Compare to MIRT - using gpuArrays
        tstart = tic;
        output_gpuarray = mirt_dctn(input_gpu);
        dt_mirt_gpu(t) = toc(tstart);
    end
        
    for t = 1:100
        % Compare to FFTW implementation
        tstart = tic;
        output = matlab_fftw(input, 'forward');
        dt_fftw(t) = toc(tstart);
    end

    % Close the libs
    matlab_cudct_static();
    matlab_cudct_gpuarray();

    % Report
    fprintf('Execution time (N, d) = (%d, %d) - numel=%d:\n', N, d, numel(input));
    fprintf('CUDCT (incl host-device transfer): %.5f (%.5f)\n', mean(dt_cudct(:)), std(dt_cudct(:)));
    fprintf('CUDCT (static, incl. host-device transfer): %.3f (%.3f)\n', mean(dt_cudct_static(:)), std(dt_cudct_static(:)));
    fprintf('CUDCT (static, w/o host-device transfer): %.5f (%.5f)\n', mean(dt_cudct_static_gpu(:)), std(dt_cudct_static_gpu(:)));
    
    fprintf('MIRT: %.5f (%.5f)\n', mean(dt_mirt(:)), std(dt_mirt(:)));
    fprintf('MIRT (gpuArray): %.5f (%.5f)\n', mean(dt_mirt_gpu(:)), std(dt_mirt_gpu(:)));
    
    fprintf('FFTW: %.4f (%.5f)\n', mean(dt_fftw(:)), std(dt_fftw(:)));

 end

