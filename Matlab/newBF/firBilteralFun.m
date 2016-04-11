function w = firBilteralFun(G_big, G , l, thres)
%FIRBILTERALFUN Summary of this function goes here
%   Detailed explanation goes here
w = zeros(size(G_big, 2), 1) + thres;
%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'off';
options.numDiff = false;
lambda = 50;

%% for check
% numgrad = computeNumericalGradient(@(w)BilteralFunCost(w, G_big, G(:, 1 : m), l, lambda, thres), w);
% [cost, grad] = BilteralFunCost(w, G_big, G(:, 1 : m), l, lambda, thres);
% disp([numgrad grad]); 

w = minFunc(@(w)BilteralFunCost(w, G_big, G, l, lambda, thres), w, options);

end

