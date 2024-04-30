% Binary Image Steganography
clc;
clear;
close all;

% Read Image as Cover Medium
img = imread("images/PeppersRGB.bmp"); % Read Image
img = rgb2gray(img); % Convert RGB to Gray
img = imbinarize(img); % Binarize

% payload: uint32
payload = uint32(2112515); % Let's say my student ID: 2112515

% You can also get payload from user input:
% payload = input("Please input your payload number (uint32): ");
% payload = uint32(payload); % Convert double type to uint32

% Insert payload
[n_rows, n_cols] = size(img);
patch_len = 4; % patch size is 1x4

% Check capacity
assert((floor(n_cols/patch_len) * n_rows) >= 32, 'The capacity of cover image is NOT enough!');

nbit = 0;
terminated = false;
img_with_info = img;
for i = 1:n_rows
    for j = 1:patch_len:(n_cols - patch_len + 1)
        n_blk = calc_black_num(img(i, j:(j+3))); % Number of black
        nbit = nbit + 1;
        curr_bit = bitget(payload, nbit); % Current payload bit

        if curr_bit == 0 % Black
            if n_blk == 1 % flip two white(1) bits
                img_with_info(i, j:(j+3)) = flip_bits(img(i, j:(j+3)), 1, 2);
            elseif n_blk == 2 % flip one white(1) bit
                img_with_info(i, j:(j+3)) = flip_bits(img(i, j:(j+3)), 1, 1);
            elseif n_blk == 4 % randomly flip one bit
                idx = round(rand(1, 1) * 3); % random number in {0, 1, 2, 3}
                img_with_info(i, j+idx) = 1;
            elseif n_blk == 0 % Invalid patch: discard
                nbit = nbit - 1; % BackTracking
            end
        else % White
            if n_blk == 3 % flip two black(0) bits
                img_with_info(i, j:(j+3)) = flip_bits(img(i, j:(j+3)), 0, 2);
            elseif n_blk == 2 % flip one black(0) bit
                img_with_info(i, j:(j+3)) = flip_bits(img(i, j:(j+3)), 0, 1);
            elseif n_blk == 0 % randomly flip one bit
                idx = round(rand(1, 1) * 3); % random number in {0, 1, 2, 3}
                img_with_info(i, j+idx) = 0;
            elseif n_blk == 4 % Invalid patch: discard
                nbit = nbit - 1; % BackTracking
            end
        end
        
        if nbit == 32
            terminated = true;
            break
        end
    end
    if terminated
        break
    end
end

assert(nbit == 32, 'The number of valid patches is NOT enough!');


% Extract payload
nbit = 0;
info = uint32(0);
terminated = false;
for i = 1:n_rows
    for j = 1:patch_len:(n_cols - patch_len + 1)
        n_blk = calc_black_num(img_with_info(i, j:(j+3)));
        if n_blk == 3
            nbit = nbit + 1;
            info = bitset(info, nbit, 0);
        elseif n_blk == 1
            nbit = nbit + 1;
            info = bitset(info, nbit, 1);
        end

        if nbit == 32
            terminated = true;
            break
        end
    end

    if terminated
        break
    end
end


% Display Results
figure(1);
subplot(1, 2, 1);
imshow(img, []);
title('Cover Image');
subplot(1, 2, 2);
imshow(img_with_info, []);
title('Image with Info');

fprintf('Hidden Message: %u\n', info);
