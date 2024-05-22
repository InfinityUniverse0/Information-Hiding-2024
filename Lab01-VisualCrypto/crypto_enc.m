function [key1,key2] = crypto_enc(img)
%CRYPTO_ENC Visual Crypto Encryption
%   可视密码 (2,2)门限方案 加密函数

scalar = 255; % 也可以修改为 1 或 scalar = max(img(:));
[m,n] = size(img);

% 原像素拓展为 2x2 个像素，共计 6 种待选图案(注意相邻的两个图案互补)
p(:,:,1) = [1 1; 0 0];
p(:,:,2) = [0 0; 1 1];
p(:,:,3) = [1 0; 1 0];
p(:,:,4) = [0 1; 0 1];
p(:,:,5) = [1 0; 0 1];
p(:,:,6) = [0 1; 1 0];

key1 = uint8(zeros(2*size(img)));
key2 = uint8(zeros(2*size(img)));

for i=1:m
    for j=1:n
        idx = randi(6);
        key1(2*i-1:2*i, 2*j-1:2*j) = p(:,:,idx);
        if img(i,j)==0
            % Black
            key2(2*i-1:2*i, 2*j-1:2*j) = not(p(:,:,idx)); % 取反
        else
            % White
            idx_out = random_choice_white_pattern(idx);
            key2(2*i-1:2*i, 2*j-1:2*j) = p(:,:,idx_out);
        end
    end
end

key1 = key1 * scalar;
key2 = key2 * scalar;
end

function idx_out = random_choice_white_pattern(idx)
% 根据key1的白像素图案，随机选择符合规定的key2的白像素图案
if bitget(idx,1)==0
    idx_pair = idx - 1;
else
    idx_pair = idx + 1;
end
while true
    idx_out = randi(6);
    if idx_out~=idx && idx_out~=idx_pair
        break
    end
end
end
