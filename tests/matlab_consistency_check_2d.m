clc;
clear all;
close all;

pkg load image;
pkg load signal;

addpath("mirt_dctn");
addpath("mex");

%% Load image volume
img = phantom();
img = imresize(img, [253, 201]);

%% Calculate the forward DCT
dct_img_matlab = dct2(img); 
dct_img_mirt = mirt_dctn(img);
dct_img_cuda = cudct(img, 'forward');
dct_img_fftw = fftw(img, 'forward');

%% Check differences between DCT transforms
diff_dct_matlab_mirt = sum((dct_img_matlab(:) - dct_img_mirt(:)).^2);
diff_dct_matlab_fftw = sum((dct_img_matlab(:) - dct_img_fftw(:)).^2);
diff_dct_matlab_cuda = sum((dct_img_matlab(:) - dct_img_cuda(:)).^2);
diff_dct_mirt_fftw = sum((dct_img_mirt(:) - dct_img_fftw(:)).^2);

fprintf("Difference DCT standard - mirt:\t%.10f\n", diff_dct_matlab_mirt);
fprintf("Difference DCT standard - fftw:\t%.10f\n", diff_dct_matlab_fftw);
fprintf("Difference DCT standard - CUDA:\t%.10f\n", diff_dct_matlab_cuda);
fprintf("Difference DCT mirt - fftw:\t%.10f\n", diff_dct_mirt_fftw);

%% Calculate the inverse DCT
recon_img_matlab = idct2(dct_img_matlab);
recon_img_mirt = mirt_idctn(dct_img_mirt);
recon_img_cuda = cudct(dct_img_cuda, 'inverse');
recon_img_fftw = fftw(dct_img_fftw, 'inverse');

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
