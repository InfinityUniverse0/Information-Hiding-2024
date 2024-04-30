function flipped_patch = flip_bits(patch, bit_type, n)
%FLIP_BITS Flip `n` bit(s) of given binary matrix `patch`.
%   bit_type is the bit type (0 or 1) to be flipped

[n_rows, n_cols] = size(patch);

count = 0;
flipped_patch = patch;
terminated = false;
for i = 1:n_rows
    for j = 1:n_cols
        if patch(i, j) == bit_type
            flipped_patch(i, j) = ~bit_type;
            count = count + 1;
        end
        if count == n
            terminated = true;
            break
        end
    end
    if terminated
        break
    end
end

assert(count == n, 'No enough bits to flip!');

end

