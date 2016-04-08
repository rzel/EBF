function [A_out]=A_vec2_matrix_tune(A)

num=size(A,1);

A_out=zeros(3,3,num);

A_out(1,1,:) = A(:,1);
A_out(1,2,:) = A(:,2);
A_out(2,1,:) = A(:,3);
A_out(2,2,:) = A(:,4);
A_out(3,3,:) = 1;

%for i=1:num
%    A_out(1,1,i)=A(i,1);
%    A_out(1,2,i)=A(i,2);
%    A_out(2,1,i)=A(i,3);
%    A_out(2,2,i)=A(i,4);
%    A_out(3,3,i)=1;
%end;


% for i=1:num
%     A_out(2,2,i)=A(i,1);
%     A_out(1,2,i)=A(i,2);
%     A_out(2,1,i)=A(i,3);
%     A_out(1,1,i)=A(i,4);
%     A_out(3,3,i)=1;
% end;

% for i=1:num
%     A_out(1,1,i)=A(i,1);
%     A_out(1,2,i)=A(i,3);
%     A_out(2,1,i)=A(i,2);
%     A_out(2,2,i)=A(i,4);
%     A_out(3,3,i)=1;
% end;

