clc;
clear all;
close all;

addpath("mirt_dctn");

%% Load image volume
img = phantom();

%% Calculate the forward DCT
dct_img_matlab_standard = dct2(img); 

%% Select index and show
p = 102;
q = 184;

fprintf("DCT (with Matlab's dct2) (%d,%d) = (%.10f)\n", p, q, dct_img_matlab_standard(p,q));

%% Calculate element by hand
val = 0;

M = size(img, 1);
N = size(img, 2);

for m = 0:M-1
    for n = 0:N-1
        val = val + img(m+1,n+1) * cos(pi * (2*m+1) * (p - 1) / (2*M)) * cos(pi * (2*n+1) * (q - 1) / (2*N) );
    end
end
val = val / sqrt(M * N);

if (q ~= 1)
    val = val * sqrt(2);
end
if (p ~= 1)
    val = val * sqrt(2);
end

fprintf("DCT (by hand) (%d, %d) = (%.10f)\n", p, q, val);
