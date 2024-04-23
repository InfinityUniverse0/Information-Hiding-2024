% Steganography using Parity Bit
clear;
clc;
close all;

% Cover Image
cover_img = imread("images/PeppersRGB.bmp"); % Read image
cover_img = rgb2gray(cover_img); % Convert RGB to Gray

% Payload Image
payload = imread("images/MisakaMikoto.png"); % Read image
payload = rgb2gray(payload); % Convert RGB to Gray
payload = imbinarize(payload); % Binarization

% Get Image Size
[cover_rows, cover_cols] = size(cover_img);
[payload_rows, payload_cols] = size(payload);

% Calculate Patch Size
patch_rows = floor(cover_rows / payload_rows);
patch_cols = floor(cover_cols / payload_cols);
assert((patch_rows > 0) && (patch_cols > 0), "Cover image capacity is NOT enough to insert payload!");

% Insert Payload
img_with_info = cover_img;
for i = 1:payload_rows
    for j = 1:payload_cols
        idx_i = (i - 1) * patch_rows + 1;
        idx_j = (j - 1) * patch_cols + 1;
        parity = calc_parity(cover_img(idx_i:(idx_i+patch_rows-1), idx_j:(idx_j+patch_cols-1)));
        if parity ~= payload(i, j)
            % Flip the LSB of the top-left element of current patch
            img_with_info(idx_i, idx_j) = bitxor(img_with_info(idx_i, idx_j), 1);
            % The line above is equivalent to the lines below
            % bit = bitget(cover_img(idx_i, idx_j), 1); % Get LSB of the top-left element of current patch
            % bit = bitxor(bit, 1); % Xor with 1 (i.e. flip the bit) 
            % img_with_info(idx_i, idx_j) = bitset(img_with_info(idx_i, idx_j), 1, bit);
        end
    end
end

% Extract Payload
info = uint8(zeros(payload_rows, payload_cols));
for i = 1:payload_rows
    for j = 1:payload_cols
        idx_i = (i - 1) * patch_rows + 1;
        idx_j = (j - 1) * patch_cols + 1;
        info(i, j) = calc_parity(img_with_info(idx_i:(idx_i+patch_rows-1), idx_j:(idx_j+patch_cols-1)));
    end
end


% Display
figure;
subplot(2, 2, 1);
imshow(cover_img, []); % Normalize and show
title("Cover Image");

subplot(2, 2, 2);
imshow(payload, []); % Normalize and show
title("Payload Image");

subplot(2, 2, 3);
imshow(img_with_info, []); % Normalize and show
title("Image with Info");

subplot(2, 2, 4);
imshow(info, []); % Normalize and show
title("Extracted Info");
