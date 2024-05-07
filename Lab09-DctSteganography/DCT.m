% DCT Steganography

clc;
clear;
close all;

% Image Reading and Preprocessing
cover_img = imread("images/PeppersRGB.bmp"); % Cover Image
cover_img = rgb2gray(cover_img); % Convert RGB to Gray
cover_img = double(cover_img) / 255; % Normalization
payload = imread("images/MisakaMikoto.png"); % Payload
payload = rgb2gray(payload); % Convert RGB to Gray
payload = imbinarize(payload); % Binarization

% Config
blk_size = 8;
a = 0.01; % alpha
[m, n] = size(cover_img);
[n_rows, n_cols] = size(payload);

assert(((m >= (n_rows * blk_size))) && ((n >= (n_cols * blk_size))), 'Cover Medium is NOT large enough!');

% Insert
img_with_info = zeros([m, n]);
for i = 1:n_rows
    for j = 1:n_cols
        x = (i - 1) * blk_size + 1;
        y = (j - 1) * blk_size + 1;
        img_with_info(x:(x+blk_size-1), y:(y+blk_size-1)) = dct2(cover_img(x:(x+blk_size-1), y:(y+blk_size-1))); % DCT-2D
        value = payload(i, j) - (~payload(i, j)); % insert 1 if payload bit is 1; else insert -1
        img_with_info(x, y) = img_with_info(x, y) * (1 + a * value);
        img_with_info(x:(x+blk_size-1), y:(y+blk_size-1)) = idct2(img_with_info(x:(x+blk_size-1), y:(y+blk_size-1))); % Inverse DCT-2D
    end
end

% Extract
info = zeros([n_rows, n_cols]);
for i = 1:n_rows
    for j = 1:n_cols
        x = (i - 1) * blk_size + 1;
        y = (j - 1) * blk_size + 1;
        info(i, j) = img_with_info(x, y) > cover_img(x, y);
    end
end

% Display
figure();
subplot(2, 2, 1);
imshow(cover_img, []);
title('Cover Image');
subplot(2, 2, 2);
imshow(payload, []);
title('Payload');
subplot(2, 2, 3);
imshow(img_with_info, []);
title('Image with Info');
subplot(2, 2, 4);
imshow(info, []);
title('Extracted Info');
