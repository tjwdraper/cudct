clc;
clear all;
close all;


addpath("mirt_dctn")
addpath("mex");

d = 3;

precision = "double"; % Either double or single

for N = [32, 64, 128, 256]

    % Create random array of dimension d
    input = randn(N*ones(d,1));
    if strcmp(precision, "single")
        input = single(input);
    endif

    % Initialize static implementation by passing expect input dimensions of size (Nx1) and precision
    octave_cudct_static(size(input)', precision);

    % Execute 100 DCT transforms
    dt_static = zeros(100, 1);
    dt_standard = zeros(100, 1);

    for t = 1:100
        % Static implementation
        tstart = cputime;
        dct_static = octave_cudct_static(input, "forward");
        dt_static(t) = cputime - tstart;

        % Standard implementation
        tstart = cputime;
        dct_standard = octave_cudct(input, "forward");
        dt_standard(t) = cputime - tstart;
    endfor

    % Close the lib
    octave_cudct_static();

    % Report
    fprintf("Execution time (N, d) = (%d, %d) - numel=%d:\n", N, d, numel(input));
    fprintf("CUDA standard: %.5f (%.5f)\n", mean(dt_standard(:)), std(dt_standard(:)));
    fprintf("CUDA static: %.5f (%.5f)\n\n", mean(dt_static(:)), std(dt_static(:)));

endfor

