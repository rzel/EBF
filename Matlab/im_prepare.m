function [im]=im_prepare(im_path, r_size)
% resizes images

if nargin == 1
    r_size = 480;
end;

im = imread(im_path);
[row, col] = size(im);
factor = min(row, col) / r_size;
im = imresize(im, 1/factor, 'bilinear');

end


