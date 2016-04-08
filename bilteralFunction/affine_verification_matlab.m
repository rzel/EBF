function [inliers]=affine_verification_matlab(f1,f2,T_a, T_b,f11, f22, thres_spatial)

thres_grad=0.1;

spatial_s=1;

f1=cat(1, f1(1:2,:), ones(1,size(f1,2)));
f2=cat(1, f2(1:2,:), ones(1,size(f2,2)));

f1=T_a*f1;
f2=T_b*f2;

f11=cat(1, f11(1:2,:), ones(1,size(f11,2)));
f22=cat(1, f22(1:2,:), ones(1,size(f22,2)));

f11=T_a*f11;
f22=T_b*f22;

Vec=cat(2, f1(1:2,:)', (f2(1:2,:)- f1(1:2,:))');

if size(f1,2)> 300
    
    ind=randsample(size(f1,2), 300);
    Vec=Vec(ind,:);
    % bilateral affine curve computed here
    [wx,ax0, ~]=grad_desc_affine(Vec, f2(1,ind),spatial_s, 1, ones(length(ind),1), thres_grad);
    [wy,ay0, ~]=grad_desc_affine(Vec, f2(2,ind),spatial_s, 1, ones(length(ind),1), thres_grad);
else
    
    [wx,ax0, ~]=grad_desc_affine(Vec, f2(1,:),spatial_s, 1, ones(size(f1,2),1), thres_grad);
    [wy,ay0, ~]=grad_desc_affine(Vec, f2(2,:),spatial_s, 1, ones(size(f1,2),1), thres_grad);
    
end;


% verfication of points using the curve is done here
[inliers]=affine_inliers(f11, f22, Vec, wx, wy, ax0, ay0, thres_spatial);


