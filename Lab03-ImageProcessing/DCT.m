% Discrete Consine Transform (DCT)
clear;
clc;

% Read Image
img = imread('images/LenaRGB.bmp');
% Convert RGB to Gray
img = rgb2gray(img);
% Binarize
I = imbinarize(img);

figure(1);
imshow(img);
title('(a) 原图像');

c = dct2(I); % Discrete Consine Transform (DCT)

figure(2);
imshow(c);
title('(b) DCT变换系数');

figure(3);
mesh(c);
title('(c) DCT变换系数(立体视图)');
