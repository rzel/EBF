function [ matches_i_all ] = filter_matches_tune(f1, f2, matches, matches_all, thres_spatial, prob_t)
% same as filter _matches but with a change to prob_t values and unique matches after
flow_mag=1;
max_prob=1;

% compute the feature affines
[Ay2x] = getA_s_y2x_new(f1(3:6, matches(1,:))', f2(3:6, matches(2,:))');
discard=clean_Ay2x(Ay2x, 5); % remove badly scaled affine from consideration
Ay2x(discard,:)=[];
matches(:,discard)=[];


matches_old=matches;
Ay2x_old=Ay2x;
if size(matches,2) > 1000 %if there are two many potential matches, sub-sample
    ind=randsample(size(matches,2), 1000);
    matches=matches(:,ind);
    Ay2x=Ay2x(ind,:);
end;

% fit a bilateral likelihoood curve to the matches, 
%curve is represented by w and Vec
[T_a, T_b, w, Vec]=matches2points_affine_reguess(f1(1:2,matches(1,:))' , f2(1:2,matches(2,:))', ...
    Ay2x, 0.1); 

% verify all potential matches using the likelihood curve, output is given
% in inliers
[inlier]=inlier_estimate_reguess(T_a, T_b,  f1(1:2,matches_old(1,:)), f2(1:2, matches_old(2,:)),Ay2x_old, w, Vec, 0.1,0.5, max_prob, flow_mag);


% apply the likelihood curve to all matching points (matches_all), output
% is given in inliers_all, with the corresponding matching index in
% matches_i_all
[Ay2x_a] = getA_s_y2x_new(f1(3:6, matches_all(1,:))', f2(3:6, matches_all(2,:))');
discard=clean_Ay2x(Ay2x_a, 5);
Ay2x_a(discard,:)=[];
matches_all(:,discard)=[];
[inlier_all]=inlier_estimate_reguess(T_a, T_b,  f1(1:2,matches_all(1,:)), f2(1:2, matches_all(2,:)),Ay2x_a, w, Vec, 0.1,prob_t, max_prob, flow_mag);
matches_i_all=matches_all(:, inlier_all);


% prepare to compute affine bilateral function
% this takes the input from verified matches in inlier
matches_i=matches_old(:, inlier);
matches_v=unique_match2(matches_i);

matches_i_all=cat(2, matches_i, matches_i_all);
matches_i_all=unique(matches_i_all', 'rows')';

% compute the bilateral function with input from matches_v
% the function verifies matches_i_all
aff_verify=affine_verification_matlab(f1(1:2, matches_v(1,:)),f2(1:2, matches_v(2,:)),T_a, T_b,f1(1:2, matches_i_all(1,:)),f2(1:2, matches_i_all(2,:)), thres_spatial);


matches_i_all=matches_i_all(:, aff_verify);

    



