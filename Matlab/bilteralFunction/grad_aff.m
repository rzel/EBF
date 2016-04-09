function [x,res]=grad_aff(M_a, N_a, thres_a,  G_a,  lambda_a)

global M;
global N;
global thres;
global G;
global lambda;
global num;
num=size(G_a,2);

M=M_a;
N=N_a;
thres=thres_a;
G=G_a;
lambda=lambda_a;


big_M=cat(1, M, sqrt(lambda)*cat(2,spblkdiag(G,G,G), zeros(3*num,3)));

big_N=cat(1, N, zeros(3*num,1));


xo=big_M\big_N;
options = optimset('Jacobian','on', 'display', 'off');

x = lsqnonlin(@myfun,xo, [], [], options);

res=M*x-N;
end

function [e, J]=myfun(x)

global M;
global N;
global thres;
global G;
global lambda;
global num;


e1=M*x-N;
e_sign1=sign(e1);
mask1=abs(e1)> thres;
e1(mask1)=sqrt(2*thres*abs(e1(mask1))-thres^2);

e2=[G*x(1:num);G*x(num+1:2*num);G*x(2*num+1:3*num)];


J1=M;
J1(mask1,:)=0.5*2*thres*diag(1./(e1(mask1).*e_sign1(mask1)))*M(mask1,:);

J2=sqrt(lambda)*cat(2,spblkdiag(G,G,G), zeros(3*num,3));
e2=sqrt(lambda)*e2;


e=cat(1, e1, e2);

J=sparse(cat(1, J1, J2));
end


