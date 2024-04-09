% LSB Steganography
clear;
clc;
close all;

cover_img = imread("images/LenaRGB.bmp"); % Cover Medium
cover_img = im2gray(cover_img); % Convert RGB to Gray
hidden_info = imread("images/MisakaMikoto.png"); % Payload (Hidden Data)
hidden_info = im2gray(hidden_info); % Convert RGB to Gray
hidden_info = imbinarize(hidden_info); % Binarization

% Here I only consider the case where both images are the same size!
assert(isequal(size(cover_img), size(hidden_info)), "Size of Images is NOT equal!");

% Insert into LSB
img_with_info = bitset(cover_img, 1, hidden_info);
% imwrite(img_with_info, "images/img_with_info.jpeg"); % Save Image

% Extract from LSB
info = bitget(img_with_info, 1);

figure(1);
subplot(2, 2, 1);
imshow(cover_img, []);
title("Cover Medium Image");
subplot(2, 2, 2);
imshow(hidden_info, []);
title("Hidden Image (Payload)");
subplot(2, 2, 3);
imshow(img_with_info, []);
title("Image with Info");
subplot(2, 2, 4);
imshow(info, []);
title("Extracted Info");

% Insert Student ID into LSB
% Assume that ID is uint32 (4 Bytes or 32 Bits)
id = input("Please input your student ID number (can fit into uint32): ");
id = uint32(id); % Convert double type to uint32

% id = uint32(2112515); % You can also use hard coding
img_with_id = cover_img;
[n_row, n_col] = size(img_with_id);

% Assume that n_col >= 32
assert(n_col >= 32, "n_col should NOT be less than 32!");

% Insert into LSB
for i = 1:32
    img_with_id(1, i) = bitset(img_with_id(1, i), 1, bitget(id, i));
end

% Extract from LSB
extracted_id = uint32(0);
for i = 1:32
    extracted_id = bitset(extracted_id, i, bitget(img_with_id(1, i), 1));
end

figure(2);
subplot(1, 2, 1);
imshow(cover_img, []);
title("Cover Medium Image");
subplot(1, 2, 2);
imshow(img_with_id, []);
title("Image with ID");

fprintf('Your Student ID Number is: %u\n', extracted_id);
