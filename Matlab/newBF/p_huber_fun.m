function [ cost, grad ] = p_huber_fun( w,  thres )
%LIKEHOODCOST Summary of this function goes here
%   Detailed explanation goes here

cost = p_huber_cost(w, thres);

grad = p_huber_grad(w, thres);
grad = grad';

end


function error = p_huber_cost(x, thres)
   e = thres^2 * (sqrt(1 + (x / thres).^2) - 1);
   error = sum(e);
end

function grad = p_huber_grad(x, thres)
   grad = x ./ sqrt(1 + (x / thres).^2);
end

