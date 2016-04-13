function [ cost, grad ] = BilteralFunCost( w, G_big, G, l, lambda, thres )
%BILTERALFUNCOST Summary of this function goes here
%   Detailed explanation goes here
m = size(G, 1);
W = reshape(w, m + 1, 3);
W(m+1, :) = [];

z = l - G_big * w;
cost = p_huber_cost(z, thres) + sum(diag(lambda * W' * G * W));
cost = cost / m;


grad = p_huber_grad(z, thres) * (-G_big);
%Wg = cat(2, lambda * 2 * W' * G, zeros(3, 1));
Wg = cat(1, lambda * 2 * G * W, zeros(1, 3)); 
grad = grad' + reshape(Wg, length(w), 1);
grad = grad / m;


end

function error = huber_cost(x, thres)
    e = zeros(length(x), 1);
    p1 = abs(x) <= thres;
    p2 = abs(x) > thres;
    e(p1) = x(p1).^2;
    e(p2) = abs(x(p2)) * 2 * thres - thres^2;
    error = sum(e);
end

function grad = huber_grad(x, thres)
    grad = zeros(1, length(x));
    p1 = abs(x) <= thres;
    grad(p1) = 2 * x(p1);
    p2 = x > thres;
    grad(p2) = 2 * thres;
    p3 = -x > thres;
    grad(p3) = -2 * thres;
end

function error = p_huber_cost(x, thres)
   e = thres^2 * (sqrt(1 + (x / thres).^2) - 1);
   error = sum(e);
end

function grad = p_huber_grad(x, thres)
   grad = x ./ sqrt(1 + (x / thres).^2);
   grad = grad';
end