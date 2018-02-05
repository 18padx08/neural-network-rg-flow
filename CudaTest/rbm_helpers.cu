
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "utils.cuh"
#include "rbm_helpers.cuh"

using namespace utils;

__global__
void contrastive_divergence(curandState *globalState,int *input, double *weights, double *bh, double *bv, bool *mask, double *ph_mean, int *ph_sample, double *nv_means, int *nv_samples, double *nh_means, int *nh_samples, int n_hidden, int n_visible, double lr, int N) {
	//printf("in cuda kernel");
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > n_visible || col > n_hidden) return;

	/* CD-k */
	//mtx.lock();
	//printf("start CD \n");

		curandState state = globalState[row*blockDim.y * 32 + col];
	sample_h_given_v(state,input, ph_mean, ph_sample, bh, weights, n_hidden, n_visible);
	__syncthreads();
	//sample_v_given_h(state, ph_sample, nv_means, nv_samples, weights, bv, n_visible, n_hidden);
	for (int step = 0; step<1; step++) {
		if (step == 0) {
			//printf("only do it once \n");
			gibbs_hvh(state,ph_sample, nv_means, nv_samples, nh_means, nh_samples, weights, bv, bv, n_visible, n_hidden);
		}
		
	}
	__syncthreads();
	
	if (mask[row*n_hidden + col]) {
		//printf("yes it was masked \n");
		//printf("%d, ", ph_mean[i]); printf("%d, ", input[j]); printf("%d, ", nh_means[i]); printf("%d, ", nv_samples[j]);
		weights[row*n_hidden + col] += lr * (ph_mean[col] * input[row] - nh_means[col] * nv_samples[row]) / N;
		//bh[col] += lr * (ph_sample[col] - nh_means[col]) / N;
		//bv[row] += lr * (input[row] - nv_samples[row]) / N;
		//printf("|%d|", weights[i*n_visible + j]);
	}
	else {
		//penalty if not masked
		//weights[row*n_hidden + col] += lr * lr * lr* (ph_mean[col] * input[row] - nh_means[col] * nv_samples[row]) / N;
		//bh[col] += lr *lr * (ph_sample[col] - nh_means[col]) / N;
		//bv[row] += lr*lr * (input[row] - nv_samples[row]) / N;
	}
}

__device__
void sample_h_given_v(curandState globalState,int *v0_sample, double *mean, int *sample, double *hbias, double *weights, int n_hidden, int n_visible) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > n_visible || col > n_hidden) return;
		double val = propup(v0_sample, weights, hbias[col], n_visible, n_hidden);
		mean[col] = val;
		int state = binomial(1, mean[col], globalState);
		sample[col] = state;
		//printf("%f -> %d\n",val,state);
	
}

__device__
void sample_v_given_h(curandState globalState,int *h0_sample, double *mean, int *sample, double *weights, double *vbias, int n_visible, int n_hidden) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > n_visible || col > n_hidden) return;

	mean[row] = propdown(h0_sample, vbias[row],n_hidden,n_visible,weights);
	sample[row] = binomial(1, mean[row], globalState);
	
}

__device__
double propup(int *v, double *w, double bh, int n_visible, int n_hidden) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > n_visible || col > n_hidden) return;

	double pre_sigmoid_activation = 0.0;
	for (int j = 0; j<n_visible; j++) {
		pre_sigmoid_activation +=v[j] * w[j*n_hidden + col];
	}
	//pre_sigmoid_activation += bh;
	return sigmoid(pre_sigmoid_activation);
}

__device__
double propdown(int *h, double bv, int n_hidden, int n_visible, double *W) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > n_visible || col > n_hidden) return;
	double pre_sigmoid_activation = 0.0;
	for (int j = 0; j<n_hidden; j++) {
		pre_sigmoid_activation += W[row*n_hidden + j] * h[j];
	}
	//pre_sigmoid_activation += bv;
	return sigmoid(pre_sigmoid_activation);
}

__device__
void gibbs_hvh(curandState globalState,int *h0_sample, double *nv_means, int *nv_samples, double *nh_means, int *nh_samples, double *weights, double *vbias, double *hbias, int n_visible, int n_hidden) {
	sample_v_given_h(globalState,h0_sample, nv_means, nv_samples, weights, vbias, n_visible, n_hidden);
	__syncthreads();
	sample_h_given_v(globalState,nv_samples, nh_means, nh_samples, hbias, weights, n_hidden, n_visible);
}
