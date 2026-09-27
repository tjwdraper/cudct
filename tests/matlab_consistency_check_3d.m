clc;
clear all;
close all;

pkg load signal;

addpath("mirt_dctn");
addpath("mex")

%% Load image volume
img = randn(101, 67, 245);

%% Calculate the forward DCT 
dct_img_mirt = mirt_dctn(img);
dct_img_cuda = cudct(img, 'forward');

dct_img_fftw = fftw_dct3(img, 'forward');

[M,N,K] = size(img);
scale = ones(M,N,K) / (8 * sqrt(M*N*K));

scale(2:end,:,:) = scale(2:end,:,:) * sqrt(2);
scale(:,2:end,:) = scale(:,2:end,:) * sqrt(2);
scale(:,:,2:end) = scale(:,:,2:end) * sqrt(2);

dct_img_fftw = dct_img_fftw .* scale;

%% Check differences between DCT transforms
diff_dct_mirt_fftw = sum((dct_img_fftw(:) - dct_img_mirt(:)).^2);
diff_dct_mirt_cuda = sum((dct_img_mirt(:) - dct_img_cuda(:)).^2);
diff_dct_fftw_cuda = sum((dct_img_fftw(:) - dct_img_cuda(:)).^2);

fprintf("Difference DCT fftw - mirt:\t%.10f\n", diff_dct_mirt_fftw);
fprintf("Difference DCT mirt - cuda:\t%.10f\n", diff_dct_mirt_cuda);
fprintf("Difference DCT fftw - cuda:\t%.10f\n", diff_dct_fftw_cuda);


%% Calculate the inverse DCT
recon_img_mirt = mirt_idctn(dct_img_mirt);
recon_img_cuda = cudct(dct_img_cuda, 'inverse');

recon_img_fftw = fftw_dct3(fftw_dct3(img, 'forward'), 'inverse') / (8 * numel(img)); % Normalization factor

%% Calculate the residual error
err_mirt = sum((recon_img_mirt(:) - img(:)).^2);
err_fftw = sum((recon_img_fftw(:) - img(:)).^2);
err_cuda = sum((recon_img_cuda(:) - img(:)).^2);

fprintf("Error (mirt):\t%.10f\n", err_mirt);
fprintf("Error (fttw):\t%.10f\n", err_fftw);
fprintf("Error (cuda):\t%.10f\n", err_cuda);