
#include <curand_kernel.h>

__global__ void contrastive_divergence(curandState *globalState,int *input, double *weights, double *bh, double *bv, bool *mask, double *ph_mean, int *ph_sample, double *nv_means, int *nv_samples, double *nh_means, int *nh_samples, int n_hidden, int n_visible, double lr = 0.01, int N=1);
__device__ void sample_h_given_v(curandState globalState,int *v0_sample, double *mean, int *sample, double *hbias, double *weights, int n_hidden, int n_visible);
__device__ double propup(int *v, double *w, double b, int n_visible, int n_hidden);
__device__ double propdown(int *h, double b, int n_hidden, int n_visible, double *W);
__device__ void gibbs_hvh(curandState globalState,int *h0_sample, double *nv_means, int *nv_samples, double *nh_means, int *nh_samples, double *weights, double *vbias, double *hbias, int n_visible, int n_hidden);
