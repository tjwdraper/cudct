clc;
clear all;
close all;


addpath("mirt_dctn")
addpath("mex");

d = 3;

precision = "double";

for N = [32, 64, 128, 256]

    % Create random array of dimension d
    img = randn(N*ones(d,1));
    if strcmp(precision, "single")
        img = single(img);
    endif

    % Initialize static implementation
    cudct_static(size(img)', precision);

    % Execute 100 DCT transforms
    dt_static = zeros(100, 1);
    dt_standard = zeros(100, 1);

    for t = 1:100
        % Static implementation
        tstart = cputime;
        cudct_static(img, "forward");
        dt_static(t) = cputime - tstart;

        % Standard implementation
        tstart = cputime;
        dct_standard = cudct(img, "forward");
        dt_standard(t) = cputime - tstart;
    endfor

    % Close the lib
    cudct_static();

    % Report
    fprintf("Execution time (N, d) = (%d, %d) - numel=%d:\n", N, d, numel(img));
    fprintf("CUDA standard: %.5f (%.5f)\n", mean(dt_standard(:)), std(dt_standard(:)));
    fprintf("CUDA static: %.5f (%.5f)\n\n", mean(dt_static(:)), std(dt_static(:)));

endfor

