clc;
clear all;
close all;

addpath("mirt_dctn");

%% Load image volume
load mri;
img = double(squeeze(D));
clear D map siz;

%% Calculate the forward DCT
dct_img_matlab_standard = dct(img); 
dct_img_matlab_mirt = mirt_dctn(img);

%% Check differences between DCT transforms
diff_dct_standard_mirt = sum((dct_img_matlab_standard(:) - dct_img_matlab_mirt(:)).^2);
fprintf("Difference DCT standard - mirt:\t%.3f\n", diff_dct_standard_mirt);

%% Calculate the inverse DCT
recon_img_matlab_standard = idct(dct_img_matlab_standard);
recon_img_matlab_mirt = mirt_idctn(dct_img_matlab_mirt);

%% Calculate the residual error
err_matlab_standard = sum((recon_img_matlab_standard(:) - img(:)).^2);
err_matlab_mirt = sum((recon_img_matlab_mirt(:) - img(:)).^2);
fprintf("Error (standard):\t%.10f\n", err_matlab_standard);
fprintf("Error (mirt):\t\t%.10f\n", err_matlab_mirt);