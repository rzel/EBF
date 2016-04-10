function [matches_all, quality] = cv_match( d1, d2 )
%CPU_MATCH Summary of this function goes here
%   Detailed explanation goes here
    [matches_all, quality] = mexMatching(d1, d2);
end

