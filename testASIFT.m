clear; clc;
addpath(genpath('.'));

%%load data

%path_a ='.\data\adam1.png';
%path_b ='.\data\adam2.png';
path_a='.\data\image047.jpg';
path_b='.\data\image048.jpg';

image1 = imread(path_a);
image2 = imread(path_b);

image1 = rgb2gray(image1);
image2 = rgb2gray(image2);

img1 = single(image1);
img2 = single(image2);

%% ASIFT
% if(resize == 1) : in source code, it will become 600 x 800;
% numTiltes : ASIFT paramter, defalut is 7, but 3 is OK.
disp('start asift');
resize = 0;
numTiltes = 7;
tic();
[f1, f2, d1, d2] = ASIFT(img1, img2, numTiltes, resize);
toc();

%% matching
% flag_flann == 0 : BFMatcher     slow !
% flag_flann == 1 : FlannMatcher  efficient but not expression !
% sift_thres == 1.5 : in eccv2014 paper setting
disp('start matching');
flag_flann = 1;
sift_thres = 1.5;
tic();
[matches, matches_all] = cv_match(d1, d2, sift_thres, flag_flann);
toc();


%% draw matches
display_match_trad_enum(uint8(image1), uint8(image2), matches, f1(1:2,:),f2(1:2,:));




