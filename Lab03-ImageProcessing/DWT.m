% Discrete Wavelet Transform (DWT)
clear;
clc;

% Read Image
img = imread('images/LenaRGB.bmp');
% Convert RGB to Gray
img = rgb2gray(img);
% Binarize
a = imbinarize(img);

nbc = size(a, 1);
nbcol = nbc / 2;

[ca1, ch1, cv1, cd1] = dwt2(a, 'db4'); % 采用 Daubechies-4 小波
[ca2, ch2, cv2, cd2] = dwt2(ca1, 'db4'); % 二级小波分解

cod_ca1 = wcodemat(ca1, nbcol);
cod_ch1 = wcodemat(ch1, nbcol);
cod_cv1 = wcodemat(cv1, nbcol);
cod_cd1 = wcodemat(cd1, nbcol);

cod_ca2 = wcodemat(ca2, nbc);
cod_ch2 = wcodemat(ch2, nbc);
cod_cv2 = wcodemat(cv2, nbc);
cod_cd2 = wcodemat(cd2, nbc);

% 一级小波分解
image([cod_ca1, cod_ch1; cod_cv1, cod_cd1]);

% 二级小波分解
tt = [cod_ca2, cod_ch2; cod_cv2, cod_cd2];
tt=imresize(tt, size(ca1)); % 调整图片大小
image([tt, cod_ch1; cod_cv1, cod_cd1]);
