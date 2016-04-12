function [ cost, grad ] = likehoodCost( w, G, lambda, thres )
%LIKEHOODCOST Summary of this function goes here
%   Detailed explanation goes here

z = 1 - G * w;
cost = p_huber_cost(z, thres) + lambda * w' * G * w;
cost = cost / length(w);

grad = p_huber_grad(z, thres) * (-G) + lambda * 2 * w' * G;
grad = grad'/ length(w);

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

