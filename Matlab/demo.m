clear; clc;
addpath(genpath('.'));

%%paramater
re_clean = 1;
display = 1;

%%load data
path_a='E:/kitti/dataset/sequences/01/image_0/000000.png';
path_b='E:/kitti/dataset/sequences/01/image_0/000010.png';

%pre-process image 
res = 320;
[image1]=im_prepare(path_a,res);
[image2]=im_prepare(path_b,res);

img1 = single(image1);
img2 = single(image2);


%% ASIFT
% if(resize == 1) : in source code, it will become 600 x 800;
% numTiltes : ASIFT paramter, defalut is 7, but 3 is OK.
disp('start asift');
resize = 0;
numTiltes = 1;
tic();
[f1, f2, d1, d2] = ASIFT(img1, img2, numTiltes, resize);
toc();

%% matching
disp('start matching');
tic();
[matches_all, quality] = cv_match(d1, d2);
%matches = matches_all(:, a > 1.5);
[a, b] = sort(quality, 'descend');
matches = matches_all(:, b(1 : 10));
matches_all(:, 500 : end) = [];
toc();

%normalize points
f1(1, :) = f1(1, :) / size(img1, 2);
f1(2, :) = f1(2, :) / size(img1, 1);
f2(1, :) = f2(1, :) / size(img2, 2);
f2(2, :) = f2(2, :) / size(img2, 1);


disp('Begin our bilteral function fitting');
%% bilteral function fitting
thres_spatial = 0.01;
tic
% our filter, matches_i_all represent the cleaned correspondance
[ matches_i_all] = filter_matches_tune(f1, f2, matches, matches_all, thres_spatial, 0.6);


% second filter pass
if re_clean==1
    %if the first pass had very few matches (add more back)
    if size(matches_i_all,2)< 1000
        num_needed=1000-size(matches_i_all,2);        
        if size(matches,2)>num_needed
            ind=randsample(size(matches,2), num_needed);
            matches_i_all=cat(2, matches_i_all, matches(:,ind));
        else
            matches_i_all=cat(2, matches_i_all, matches);
        end;
    end;
    [ matches_i_all] = filter_matches_tune(f1, f2, matches_i_all, matches_all, thres_spatial, 0.7);
end;

toc

disp(strcat('number of matches:',  num2str(size(matches_i_all,2))));


f1(1, :) = f1(1, :) * size(img1, 2);
f1(2, :) = f1(2, :) * size(img1, 1);
f2(1, :) = f2(1, :) * size(img2, 2);
f2(2, :) = f2(2, :) * size(img2, 1);

if display
    display_match_trad_enum(uint8(image1), uint8(image2), matches_i_all, f1(1:2,:),f2(1:2,:));
    ShowMatches_color_split(uint8(image1),uint8(image2), f1(1,matches_i_all(1,:))',f1(2,matches_i_all(1,:))',...
        f2(1, matches_i_all(2,:))',f2(2, matches_i_all(2,:))',2, [0 0.5 1], 1, 1);
    ShowMatches_color_split(uint8(image1),uint8(image2), f1(1,matches_i_all(1,:))',f1(2,matches_i_all(1,:))',...
        f2(1, matches_i_all(2,:))',f2(2, matches_i_all(2,:))',2, [0 0.5 1], 1, 0);
   
end;







