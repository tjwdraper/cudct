clc;
clear all;
close all;

pkg load signal;

addpath("mirt_dctn");
addpath("mex");

%% Load image volume
img = randn(123,1);

%% Calculate the forward DCT
dct_img_matlab = dct(img); 
dct_img_mirt = mirt_dctn(img);
dct_img_cuda = cudct(img, 'forward');

dct_img_fftw = fftw_dct1(img, 'forward');

N = length(img);
scale = ones(N,1) / (2 * sqrt(N));

scale(2:end) = scale(2:end) * sqrt(2);

dct_img_fftw = dct_img_fftw .* scale;

%% Check differences between DCT transforms
diff_dct_matlab_mirt = sum((dct_img_matlab(:) - dct_img_mirt(:)).^2);
diff_dct_matlab_fftw = sum((dct_img_matlab(:) - dct_img_fftw(:)).^2);
diff_dct_matlab_cuda = sum((dct_img_matlab(:) - dct_img_cuda(:)).^2);
diff_dct_mirt_fftw = sum((dct_img_mirt(:) - dct_img_fftw(:)).^2);

fprintf("Difference DCT standard - mirt:\t%.10f\n", diff_dct_matlab_mirt);
fprintf("Difference DCT standard - fftw:\t%.10f\n", diff_dct_matlab_fftw);
fprintf("Difference DCT standard - cuda:\t%.10f\n", diff_dct_matlab_cuda);
fprintf("Difference DCT mirt - fftw:\t%.10f\n", diff_dct_mirt_fftw);

%% Calculate the inverse DCT
recon_img_matlab = idct2(dct_img_matlab);
recon_img_mirt = mirt_idctn(dct_img_mirt);
recon_img_cuda = cudct(dct_img_cuda, 'inverse');
recon_img_fftw = fftw_dct1(fftw_dct1(img, 'forward'), 'inverse') / (2 * numel(img)); % Normalization factor

%% Calculate the residual error
err_matlab = sum((recon_img_matlab(:) - img(:)).^2);
err_mirt = sum((recon_img_mirt(:) - img(:)).^2);
err_fftw = sum((recon_img_fftw(:) - img(:)).^2);
err_cuda = sum((recon_img_cuda(:) - img(:)).^2);

fprintf("Error (standard):\t%.10f\n", err_matlab);
fprintf("Error (mirt):\t\t%.10f\n", err_mirt);
fprintf("Error (fftw):\t\t%.10f\n", err_fftw);
fprintf("Error (cuda):\t\t%.10f\n", err_cuda);

%%
clear functions;
close all;
