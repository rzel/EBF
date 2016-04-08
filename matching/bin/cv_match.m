function [matches, matches_all] = cv_match( d1, d2, sift_thres, flag_flann )
%CPU_MATCH Summary of this function goes here
%   Detailed explanation goes here
    [matches, matches_all] = mexMatching(d1, d2, sift_thres, flag_flann);
end

