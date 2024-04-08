% Bit Plane of Image
clear;
clc;
close all;

img = imread("images/LenaRGB.bmp"); % Read Image
img = im2gray(img); % Convert RGB to Gray

% Extract bit plane from 1 to 8
figure(1);
for t = 1:8
    bit_plane = bitget(img, t); % Get the t-th bit
    subplot(2, 4, t);
    imshow(bit_plane, []); % Normalize and Display
    title(['第', num2str(t), '位平面']);
end

% Filter out lower bit planes and display the remaining higher bits plane
figure(2);
img_size = size(img);
higher_planes = img; % Higher bit planes
lower_planes = zeros(img_size); % Lower bit planes
for t = 1:7
    lower_planes = bitset(lower_planes, t, bitget(higher_planes, t));
    higher_planes = bitset(higher_planes, t, zeros(img_size));
    subplot(4, 4, 2*t-1);
    imshow(higher_planes, []); % Normalize and Display
    title(['Higher ', num2str(8-t), ' bits plane']);
    subplot(4, 4, 2*t);
    imshow(lower_planes, []); % Normalize and Display
    title(['Lower ', num2str(t), ' bits plane']);
end

% Filter out the t-th bit plane and diplay the remaining image
figure(3);
for t = 1:8
    filtered_bit_plane = bitget(img, t);
    remaining_img = bitset(img, t, zeros(img_size));
    subplot(4, 4, 2*t-1);
    imshow(remaining_img, []); % Normalize and Display
    title(['Image with ', num2str(t), '-th bit filted']);
    subplot(4, 4, 2*t);
    imshow(filtered_bit_plane, []); % Normalize and Display
    title(['The ', num2str(t), '-th bit plane']);
end
