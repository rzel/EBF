#pragma once
#include <Header.h>

class getInlier{
public:
	static void likehood(MatrixXf &w, MatrixXf &X_query, MatrixXf &matching, double inlier_threshold){
		MatrixXf G = Tools::get_GSM_fast(X_query);	
		MatrixXi inlier_idx =  ((G * w).array() > inlier_threshold).cast<int>();
		int num_inlier = inlier_idx.array().sum();

		MatrixXf temp_X_query(4, num_inlier), temp_matching(4, num_inlier);
		int cur = 0;
		// filter X_query and matching
		for (int i = 0; i < inlier_idx.size(); i++){
			if (inlier_idx(i, 0) == 0)	continue;
			memcpy(temp_X_query.data() + 4 * cur, X_query.data() + 4 * i, 4 * sizeof(float));
			memcpy(temp_matching.data() + 4 * cur, matching.data() + 4 * i, 4 * sizeof(float));
			cur++;
		}
		assert(num_inlier == cur);
		X_query = temp_X_query;
		matching = temp_matching;
	}

	static void likehood_all(MatrixXf &w, MatrixXf &X_all, MatrixXf &X_query, MatrixXf &matching_all, double inlier_threshold){
		MatrixXf G = Tools::get_GSM_fast(X_all, X_query);
		MatrixXi inlier_idx = ((G * w).array() > inlier_threshold).cast<int>();
		int num_inlier = inlier_idx.array().sum();
	
		MatrixXf temp_X_all(4, num_inlier), temp_matching(4, num_inlier);
	
		// filter X_query and matching
		int cur = 0;
		for (int i = 0; i < inlier_idx.size(); i++){
			if (inlier_idx(i, 0) == 0 )	continue;
			memcpy(temp_X_all.data() + 4 * cur, X_all.data() + 4 * i, 4 * sizeof(float));
			memcpy(temp_matching.data() + 4 * cur, matching_all.data() + 4 * i, 4 * sizeof(float));
			cur++;
		}
		assert(num_inlier == cur);
		X_all = temp_X_all;
		matching_all = temp_matching;
	}


	static void bilateral_function(MatrixXf &w1, MatrixXf&w2, MatrixXf &U, MatrixXf &X_all, MatrixXf &X_query, MatrixXf &matching_all, double bilateral_threshold){
		int m = (int)X_all.cols();
		int n = (int)U.cols();
		int N = 3 * n + 3;
	
		MatrixXf Lx = matching_all.row(2).transpose();
		MatrixXf Ly = matching_all.row(3).transpose();
		MatrixXf big_G, G;

		// construct G and G_big	
		G = Tools::get_GSM_fast(X_all, X_query) * U;
		big_G.resize(m, N);
		big_G.block(0, 0, m, n) = G.array() * (Lx * MatrixXf::Ones(1, n)).array();
		big_G.block(0, n, m, n) = G.array() * (Ly * MatrixXf::Ones(1, n)).array();
		big_G.block(0, 2 * n, m, n) = G;
		big_G.block(0, 3 * n, m, 3) = MatrixXf::Ones(m, 3);

		MatrixXf ex = Lx - big_G * w1;
		MatrixXf ey = Ly - big_G * w2;

		MatrixXi inlier_idx = ((ex.array().square() + ey.array().square()) < bilateral_threshold).cast<int>();
		int num_inlier = inlier_idx.array().sum();
//		MatrixXf  temp_matching(4, num_inlier);

		// filter X_query and matching
		int cur = 0;
		for (int i = 0; i < inlier_idx.size(); i++){
			if (inlier_idx(i, 0) == 0)	continue;
			memcpy(matching_all.data() + 4 * cur, matching_all.data() + 4 * i, 4 * sizeof(float));
			cur++;
		}
		assert(num_inlier == cur);
		matching_all = matching_all.leftCols(num_inlier);
	}


};



