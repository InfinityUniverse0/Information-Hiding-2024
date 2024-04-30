function blk_num = calc_black_num(patch)
%CALC_BLACK_NUM Calculate number of blacks of binary matrix `patch`
blk_num = numel(patch) - sum(patch(:));
end
