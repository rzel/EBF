#pragma once
#include <Header.h>

class getInlier{
public:
	static void likehood(MatrixXd &w, MatrixXd &G, MatrixXd &X_query, MatrixXd &matching, double inlier_threshold){
		MatrixXi inlier_idx = ((1 - (G * w).array()).abs() < inlier_threshold).cast<int>();
		int num_inlier = inlier_idx.array().sum();
	
		MatrixXd temp_X_query(4, num_inlier), temp_matching(4, num_inlier);

		int cur = 0;
		// filter X_query and matching
		for (int i = 0; i < inlier_idx.size(); i++){
			if (inlier_idx(i, 0) == 0)	continue;
			temp_X_query.col(cur) = X_query.col(i);
			temp_matching.col(cur) = matching.col(i);
			cur++;
		}
		assert(num_inlier == cur);
		X_query = temp_X_query;
		matching = temp_matching;
	}

	static void likehood_all(MatrixXd &w, MatrixXd &X_all, MatrixXd &X_query, MatrixXd &matching_all, double inlier_threshold){
		MatrixXd G = Tools::getGSM(X_all, X_query);
		MatrixXi inlier_idx = ((1 - (G * w).array()).abs() < inlier_threshold).cast<int>();
		int num_inlier = inlier_idx.array().sum();

		MatrixXd temp_X_all(4, num_inlier), temp_matching(4, num_inlier);

		
		// filter X_query and matching
		int cur = 0;
		for (int i = 0; i < inlier_idx.size(); i++){
			if (inlier_idx(i, 0) ==0 )	continue;
			temp_X_all.col(cur) = X_all.col(i);
			temp_matching.col(cur) = matching_all.col(i);
			cur++;
		}
		assert(num_inlier == cur);

		X_all = temp_X_all;
		matching_all = temp_matching;
	}


	static void bilateral_function(MatrixXd &w1, MatrixXd&w2, MatrixXd &X_all, MatrixXd &X_query, MatrixXd &matching_all, double bilateral_threshold){
		int m = (int)X_all.cols();
		int n = (int)X_query.cols();
		int N = 3 * n + 3;
	
		MatrixXd Lx = matching_all.row(2).transpose();
		MatrixXd Ly = matching_all.row(3).transpose();
		MatrixXd big_G, G;

		// construct G and G_big	
		G = Tools::getGSM(X_all, X_query);
		big_G.resize(m, N);
		big_G.block(0, 0, m, n) = G.array() * (Lx * MatrixXd::Ones(1, n)).array();
		big_G.block(0, n, m, n) = G.array() * (Ly * MatrixXd::Ones(1, n)).array();
		big_G.block(0, 2 * n, m, n) = G;
		big_G.block(0, 3 * n, m, 3) = MatrixXd::Ones(m, 3);

		MatrixXd ex = Lx - big_G * w1;
		MatrixXd ey = Ly - big_G * w2;

		MatrixXi inlier_idx = ((ex.array().square() + ey.array().square()) < bilateral_threshold).cast<int>();
		int num_inlier = inlier_idx.array().sum();
		MatrixXd  temp_matching(4, num_inlier);

		// filter X_query and matching
		int cur = 0;
		for (int i = 0; i < inlier_idx.size(); i++){
			if (inlier_idx(i, 0) == 0)	continue;
			temp_matching.col(cur) = matching_all.col(i);
			cur++;
		}
		assert(num_inlier == cur);
		matching_all = temp_matching;
	}


};



