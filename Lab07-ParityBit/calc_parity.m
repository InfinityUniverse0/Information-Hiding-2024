function parity = calc_parity(patch)
%CALC_PARITY Calculate the parity of the LSBs of given patch (matrix)
%   patch: matrix to calculate LSB Parity
%   parity: the parity returned (return 1 if LSBs of patch have odd number of 1s, else return 0)

patch_LSB = bitget(patch, 1); % Extract LSBs of patch
parity = sum(patch_LSB(:)); % Calculate the sum of LSBs
parity = bitget(parity, 1); % Calculate parity (same as mod(parity, 2))

end
