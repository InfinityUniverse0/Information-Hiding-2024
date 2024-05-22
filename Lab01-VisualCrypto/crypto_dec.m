function [secret_img, secret_img_2x] = crypto_dec(key1,key2)
%CRYPTO_DEC Visual Crypto Decryption
%   可视密码 (2,2)门限方案 解密函数

scalar = 255; % 也可以修改为 1 或 scalar = max(key1(:));

[m,n] = size(key1);
m = m/2;
n = n/2;

% secret image scaled
secret_img = uint8(zeros([m n]));
% secret image without scaling i.e. original 2x
secret_img_2x = bitand(key1, key2);

for i=1:m
    for j=1:n
        if sum(secret_img_2x(2*i-1:2*i,2*j-1:2*j)) == 0
            secret_img(i,j) = 0;
        else
            secret_img(i,j) = scalar;
        end
    end
end
end
