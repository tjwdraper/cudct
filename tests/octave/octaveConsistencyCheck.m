clc;
clear all;
close all;

pkg load signal;

addpath("mirt_dctn");
addpath("mex")

% Select precision
precision = "double";

% Select dimensions
dims = [101, 67, 245]; % Some arbitrary dimensions;

%% Create input data
input = randn(dims);
if (strcmp(precision, "single"))
    input = single(input);
endif

%% Calculate the forward DCT (if d = 1,2), one could additionally compare with Matlab's/GNU Octave dct/dct2 functions.
dct_mirt = mirt_dctn(input);
dct_cuda = octave_cudct(input, 'forward');
dct_fftw = octave_fftw(input, 'forward');

%% Check differences between DCT transforms
diff_mirt_fftw = sum((dct_fftw(:) - dct_mirt(:)).^2);
diff_mirt_cuda = sum((dct_mirt(:) - dct_cuda(:)).^2);
diff_fftw_cuda = sum((dct_fftw(:) - dct_cuda(:)).^2);

fprintf("Difference DCT fftw - mirt:\t%.10f\n", diff_mirt_fftw);
fprintf("Difference DCT mirt - cuda:\t%.10f\n", diff_mirt_cuda);
fprintf("Difference DCT fftw - cuda:\t%.10f\n", diff_fftw_cuda);

%% Calculate the inverse DCT
recon_mirt = mirt_idctn(dct_mirt);
recon_cuda = octave_cudct(dct_cuda, 'inverse');
recon_fftw = octave_fftw(dct_fftw, 'inverse');

%% Calculate the residual error
residual_mirt = sum((recon_mirt(:) - input(:)).^2);
residual_fftw = sum((recon_fftw(:) - input(:)).^2);
residual_cuda = sum((recon_cuda(:) - input(:)).^2);

fprintf("Difference recon (mirt):\t%.10f\n", residual_mirt);
fprintf("Difference recon (fttw):\t%.10f\n", residual_fftw);
fprintf("Difference recon (cuda):\t%.10f\n", residual_cuda);