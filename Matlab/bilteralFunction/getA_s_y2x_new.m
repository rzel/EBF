function [Ay2x]= getA_s_y2x_new(param1, param2)


[A1, S1] = Affine_scale_fromASIFT(param1);
[A2, S2] = Affine_scale_fromASIFT(param2);
Ay2x = zeros(size(A1, 1), 4);


for i = 1:size(Ay2x)
    Ay2x(i, :) = reshape((inv(reshape(A1(i, :), 2, 2)')*(reshape(A2(i, :), 2, 2)'))', 1, 4);
end

S = S2 ./ S1; % inverse scale
Ay2x = bsxfun(@rdivide, Ay2x, S);

