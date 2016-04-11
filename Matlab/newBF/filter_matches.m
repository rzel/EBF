function [ matches_i_all ] = filter_matches(f1, f2, matches, matches_all, thres_spatial, inlier_thres)
%FILTER_MATCHES Summary of this function goes here
%   Detailed explanation goes here


% compute the affines and clean the matches

% [Ay2x, discard] = getCleandAffine(f1(3:6, matches(1,:))', f2(3:6, matches(2,:))', 5);
% matches(:,discard)=[];
% [Ay2x_all, discard] = getCleandAffine(f1(3:6, matches_all(1,:))', f2(3:6, matches_all(2,:))', 5);
% matches_all(:,discard)=[];

% fit a likehood model
matches_old = matches;
%Ay2x_old = Ay2x;
if size(matches,2) > 200 %if there are two many potential matches, sub-sample
    ind = randsample(size(matches,2), 200);
    matches = matches(:,ind);
%    Ay2x = Ay2x(ind,:);
end;

[mu, sigma, X_L] = constructLikehoodData(f1(1:2,matches(1,:))' , f2(1:2,matches(2,:))');
w = fitLikehoodModel(X_L, 0.1);
inlier = LikehoodFilter( f1(1:2,matches_old(1,:))', f2(1:2, matches_old(2,:))', w, X_L, mu, sigma, inlier_thres);
inlier_all = LikehoodFilter( f1(1:2,matches_all(1,:))', f2(1:2, matches_all(2,:))', w, X_L, mu, sigma, inlier_thres);
matches_i_all = matches_all(:, inlier_all);



% %prepare to compute affine bilateral function
%this takes the input from verified matches in inlier
matches_v = matches_old(:, inlier);

matches_i_all = unique(matches_i_all', 'rows')';


% compute the bilateral function with input from matches_v
% the function verifies matches_i_all

f1_train = f1(1:2,matches_v(1,:))';
f2_train = f2(1:2,matches_v(2,:))';
f1_all = f1(1:2,matches_i_all(1,:))';
f2_all = f2(1:2,matches_i_all(2,:))';

inlier_all = BiFun(f1_train, f2_train, mu, sigma, f1_all, f2_all, thres_spatial);

matches_i_all = matches_i_all(:, inlier_all);


end

