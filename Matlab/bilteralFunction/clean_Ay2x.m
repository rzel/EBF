function [discard]=clean_Ay2x(Ay2x, thres)

M=sort(abs(Ay2x),2, 'descend');

M=M(:,1)./M(:,2);

discard=find(M > thres);



