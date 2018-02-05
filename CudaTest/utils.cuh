#pragma once

#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include <ctime>
using namespace std;


namespace utils {



	 __device__ double uniform(double min, double max, curandState state ) {
		 

		 double rand1 = curand_uniform_double(&state);
		// printf("%f\n", rand1);
		return rand1 * (max - min) + min;
	}

	__device__ int binomial(int n, double p, curandState state) {
		if (p < 0 || p > 1) {
			printf("wtf\n");
			return 0;
		}
		double rand1 = curand_uniform_double(&state);
		//printf("%f\n", rand1);
		if (rand1 < p) {
			return 1;
		}
		return 0;
	}

	__device__ double sigmoid(double x) {
		//printf("x: %f\n", x);
		//printf("%f", 1.0 / (1.0 + exp(-x)));
		return 1.0 / (1.0 + exp(-x));
	}

}
