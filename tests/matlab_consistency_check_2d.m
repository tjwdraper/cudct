clc;
clear all;
close all;

addpath("mirt_dctn");
addpath("mex");

%% Load image volume
img = phantom();
img = imresize(img, [253, 201]);

%% Calculate the forward DCT
dct_img_matlab_standard = dct2(img); 
dct_img_matlab_mirt = mirt_dctn(img);

dct_img_fftw = matlab_dct2(img, 'forward');

[M,N] = size(img);
scale = ones(M,N) / (4 * sqrt(M*N));

scale(2:end,:) = scale(2:end,:) * sqrt(2);
scale(:,2:end) = scale(:,2:end) * sqrt(2);

dct_img_fftw = dct_img_fftw .* scale;

%% Check differences between DCT transforms
diff_dct_standard_mirt = sum((dct_img_matlab_standard(:) - dct_img_matlab_mirt(:)).^2);
diff_dct_standard_fftw = sum((dct_img_matlab_standard(:) - dct_img_fftw(:)).^2);
diff_dct_mirt_fftw = sum((dct_img_matlab_mirt(:) - dct_img_fftw(:)).^2);

fprintf("Difference DCT standard - mirt:\t%.10f\n", diff_dct_standard_mirt);
fprintf("Difference DCT standard - fftw:\t%.10f\n", diff_dct_standard_fftw);
fprintf("Difference DCT mirt - fftw:\t%.10f\n", diff_dct_mirt_fftw);

%% Calculate the inverse DCT
recon_img_matlab_standard = idct2(dct_img_matlab_standard);
recon_img_matlab_mirt = mirt_idctn(dct_img_matlab_mirt);
recon_img_fftw = matlab_dct2(matlab_dct2(img, 'forward'), 'inverse') / (4 * numel(img)); % Normalization factor

%% Calculate the residual error
err_matlab_standard = sum((recon_img_matlab_standard(:) - img(:)).^2);
err_matlab_mirt = sum((recon_img_matlab_mirt(:) - img(:)).^2);
err_matlab_fftw = sum((recon_img_fftw(:) - img(:)).^2);

fprintf("Error (standard):\t%.10f\n", err_matlab_standard);
fprintf("Error (mirt):\t\t%.10f\n", err_matlab_mirt);
fprintf("Error (fftw):\t\t%.10f\n", err_matlab_fftw);

%%
clear functions;
close all;
