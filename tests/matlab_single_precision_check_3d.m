clc;
clear all;
close all;

pkg load signal;

addpath("mirt_dctn");
addpath("mex")

%% Load image volume
img = single(randn(101, 67, 245));

%% Calculate the forward DCT 
dct_img_mirt = mirt_dctn(img);
dct_img_cuda = cudct(img, 'forward');

%% Check differences between DCT transforms
diff_dct_mirt_cuda = sum((dct_img_mirt(:) - dct_img_cuda(:)).^2);
fprintf("Difference DCT mirt - cuda:\t%.10f\n", diff_dct_mirt_cuda);


%% Calculate the inverse DCT
recon_img_mirt = mirt_idctn(dct_img_mirt);
recon_img_cuda = cudct(dct_img_cuda, 'inverse');

%% Calculate the residual error
err_mirt = sum((recon_img_mirt(:) - img(:)).^2);
err_cuda = sum((recon_img_cuda(:) - img(:)).^2);

fprintf("Error (mirt):\t%.10f\n", err_mirt);
fprintf("Error (cuda):\t%.10f\n", err_cuda);