function [f1, f2, matches_i_all, image1, image2, d1, d2,matches_ori]=bilateral_matcher(path_a, path_b, res, thres_sift, display, thres_spatial, re_clean)


if nargin==0
    thres_sift=1.5;% for descriptors
    res=480;
    display=1;
    thres_spatial=0.01;
    re_clean=1;
    
    path_a ='data\P1011077.jpg';
    path_b ='data\P1011079.jpg';
   
end;



s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);



'Begin SIFT Matching'



% loads and resizes images
[image1]=im_prepare(path_a,res);
[image2]=im_prepare(path_b,res);

% computes the various angles at which we desire the compute SIFT
% identical to A-SIFT
[V]=tilts_sample2(7);

% SIFT matching between image pairs
% Thres is the threshold at which SIFT matches are accepted (inverse of
% lowe SIFT). we use a threshold far weaker than usual
% The squared error at a typical
% loww SIFT threshold is thres=1/(0.6^2)
[f1, d1, f2, d2, matches,matches_all, score]=match_asift_tune_complete(image1, image2, V, thres_sift);
% f1, f2: reprsent features on image1, image2
% matches: are matches above thres_sift
% matches_all: are all matches with no thresholding 


if display==2
    display_match_trad_enum(uint8(image1), uint8(image2), matches_all, f1(1:2,:),f2(1:2,:));
    ShowMatches_color_split(uint8(image1),uint8(image2), f1(1,matches(1,:))',f1(2,matches(1,:))',f2(1, matches(2,:))',f2(2, matches(2,:))',2, [0 0.5 1], 1, 1);
    ShowMatches_color_split(uint8(image1),uint8(image2), f1(1,matches(1,:))',f1(2,matches(1,:))',f2(1, matches(2,:))',f2(2, matches(2,:))',2, [0 0.5 1], 1, 0);
end;

disp('Begin our bilteral function fitting');

matches_ori=matches_all;
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

if display
   
    display_match_trad_enum(uint8(image1), uint8(image2), matches_i_all, f1(1:2,:),f2(1:2,:));
    ShowMatches_color_split(uint8(image1),uint8(image2), f1(1,matches_i_all(1,:))',f1(2,matches_i_all(1,:))',...
        f2(1, matches_i_all(2,:))',f2(2, matches_i_all(2,:))',2, [0 0.5 1], 1, 1);
    ShowMatches_color_split(uint8(image1),uint8(image2), f1(1,matches_i_all(1,:))',f1(2,matches_i_all(1,:))',...
        f2(1, matches_i_all(2,:))',f2(2, matches_i_all(2,:))',2, [0 0.5 1], 1, 0);
   
end;

return;

