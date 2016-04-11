#pragma once


class huber_function {
public:
	static double getCost(double * x, int length, double threshold) {
		double cost = 0;
		for (int i = 0; i < length; i++)
		{
			double temp = abs(x[i]) < threshold ? pow(x[i], 2) : (2 * threshold*abs(x[i]) - pow(threshold, 2));
			cost += temp;
		}
		return cost;
	}

	static void getGrad(double * x, double * g, int length, double threshold) {	
//#pragma omp parallel for
		for (int i = 0; i < length; i++)
		{
			if (abs(x[i]) < threshold) { g[i] = 2 * x[i]; }
			else if ( x[i] > threshold) { g[i] = 2 * threshold; }
			else if (-x[i] > threshold) { g[i] = -2 * threshold; }
		}
	}
};



