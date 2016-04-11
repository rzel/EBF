function w = fitLikehoodModel( X_train, thres )
%FITLIKEHOODMODEL Summary of this function goes here
%   Detailed explanation goes here

G = G_compute_fast(X_train);
m = size(G, 1);
w = zeros(m, 1) + thres;


%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
%options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'off';
options.numDiff = false;
lambda = 0.01;

%% for check
%numgrad = computeNumericalGradient(@(w)likehoodCost(w, G, lambda, thres), w);
%[cost, grad] = likehoodCost(w, G, lambda, thres);
%disp([numgrad grad]); 

w = minFunc(@(w)likehoodCost(w, G, lambda, thres), w, options);

end

