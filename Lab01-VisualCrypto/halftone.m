function im_bin = halftone(img)
%HALFTONE Halftone a gray image using Error Diffusion
%   使用误差扩散法对灰度图像进行半色调化处理
[m,n] = size(img);
for i=1:m
    for j=1:n
        if img(i,j) > 127
            out = 255;
        else
            out = 0;
        end
        err = img(i,j) - out;
        % right
        if j < n
            img(i, j+1) = img(i, j+1) + err*(7/16);
        end
        % bottom
        if i < m
            img(i+1, j) = img(i+1, j) + err*(5/16);
        end
        % bottom left
        if i<m && j>1
            img(i+1, j-1) = img(i+1, j-1) + err*(3/16);
        end
        % bottom right
        if i<m && j<n
            img(i+1, j+1) = img(i+1, j+1) + err*(1/16);
        end
        % modify center
        img(i, j) = out;
    end
end
im_bin = img;
end
