% Discrete Fourier Transform (DFT)
clear;
clc;

% Read Image
img = imread('images/LenaRGB.bmp');
% imshow(img);

% Convert RGB to Gray
figure(1);
img = rgb2gray(img);
imshow(img);
title('(a) 原图像');

figure(2);
img = imbinarize(img); % Binarize
fa = fft2(img); % FFT
ffa = fftshift(fa); % fftshift函数调整fft函数的输出顺序，将零频位置移到频谱的中心
imshow(ffa, [200,225]); % 显示灰度在200−225之间的像
title('(b) 幅度谱');

figure(3);
l = mesh(abs(ffa)); % 画网格曲面图
title("(c) 幅度谱的能量分布");
