clc;
clear all;
close all;

addpath("mirt_dctn");

%% Load image volume
load mri;
img = double(squeeze(D));
clear D map siz;

%% Calculate the forward DCT 
dct_img_matlab_mirt = mirt_dctn(img);

dct_img_fftw = matlab_dct3(img, 'forward');

[M,N,K] = size(img);
scale = ones(M,N,K) / (8 * sqrt(M*N*K));

scale(2:end,:,:) = scale(2:end,:,:) * sqrt(2);
scale(:,2:end,:) = scale(:,2:end,:) * sqrt(2);
scale(:,:,2:end) = scale(:,:,2:end) * sqrt(2);

dct_img_fftw = dct_img_fftw .* scale;

%% Check differences between DCT transforms
diff_dct_standard_fftw = sum((dct_img_fftw(:) - dct_img_matlab_mirt(:)).^2);
fprintf("Difference DCT fftw - mirt:\t%.10f\n", diff_dct_standard_fftw);

%% Calculate the inverse DCT
recon_img_matlab_mirt = mirt_idctn(dct_img_matlab_mirt);
recon_img_fftw = matlab_dct3(matlab_dct3(img, 'forward'), 'inverse') / (8 * numel(img)); % Normalization factor

%% Calculate the residual error
err_matlab_mirt = sum((recon_img_matlab_mirt(:) - img(:)).^2);
err_fftw = sum((recon_img_fftw(:) - img(:)).^2);

fprintf("Error (mirt):\t%.10f\n", err_matlab_mirt);
fprintf("Error (fttw):\t%.10f\n", err_fftw);