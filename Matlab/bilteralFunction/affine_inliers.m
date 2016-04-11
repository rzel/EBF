function [inliers]=affine_inliers(f1, f2, Vec, wx, wy, ax0, ay0, thres_spatial)


P=cat(1, f1(1:2,:), (f2(1:2,:)- f1(1:2,:)));

[G]=G_compute_fast(P', Vec, 1);


Ax=G*wx;
Ax(:,1)=Ax(:,1)+ax0(1);
Ax(:,2)=Ax(:,2)+ax0(2);
Ax(:,3)=Ax(:,3)+ax0(3);
Ay=G*wy;
Ay(:,1)=Ay(:,1)+ay0(1);
Ay(:,2)=Ay(:,2)+ay0(2);
Ay(:,3)=Ay(:,3)+ay0(3);

x=sum(Ax'.*f1(1:3,:),1);
y=sum(Ay'.*f1(1:3,:),1);

error=((x-f2(1,:)).^2+(y-f2(2,:)).^2);


mask=error< thres_spatial;
inliers=find(mask);


