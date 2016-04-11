function [ Affine, discard ] = getCleandAffine( param1, param2, thres )

% compute the affines and clean the matches
[A1, S1] = Affine_scale_fromASIFT(param1);
[A2, S2] = Affine_scale_fromASIFT(param2);
S = S2 ./ S1; % inverse scale

Affine = zeros(size(A1, 1), 4);
for i = 1:size(Affine)
    Affine(i, :) = reshape((inv(reshape(A1(i, :), 2, 2)')*(reshape(A2(i, :), 2, 2)'))', 1, 4);
end
Affine = bsxfun(@rdivide, Affine, S);


M = sort(abs(Affine),2, 'descend');
M = M(:,1)./M(:,2);
discard = M > thres;
Affine(discard, :) = [];

end


function [A, S] = Affine_scale_fromASIFT(param4)

A = zeros(size(param4, 1), 4);  
% reverse
cos_psi = cos(-param4(:, 2));% this is correct -
sin_psi = sin(-param4(:, 2));
% param4(:, 3), param4(:, 4) are from simulated image to original image
t = param4(:, 3); % this is probably correct upright
cos_phi = cos(-param4(:, 4)); % this have not checked -
sin_phi = sin(-param4(:, 4));

A(:, 1) = cos_psi.*t.*cos_phi + (-sin_psi).*sin_phi;
A(:, 2) = cos_psi.*t.*(-sin_phi) + (-sin_psi).*cos_phi;
A(:, 3) = sin_psi.*t.*cos_phi + cos_psi.*sin_phi;
A(:, 4) = sin_psi.*t.*(-sin_phi) + cos_psi.*cos_phi;
% [A(1) A(2)
% A(3) A(4)]
S=param4(:,1); 
end

