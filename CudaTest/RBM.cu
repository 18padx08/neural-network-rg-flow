
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "RBM.cuh"
#include "rbm_helpers.cuh"
#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <ctime>

__global__ void initialise_curand_on_kernels(curandState * state, unsigned long long seed);

double uniform(double min, double max) {
	return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

RBM::RBM(int n_visible, int n_hidden, bool * mask) :mean(0), mask(mask), bh((double *)malloc(n_hidden * sizeof(double))), bv((double*)malloc(n_visible * sizeof(double ))), n_hidden(n_hidden), n_visible(n_visible)
{
	weights = (double *)malloc(n_hidden * n_visible * sizeof(double));
	for (int i = 0; i < n_hidden*n_visible; i++) {
		if(mask[i])
			weights[i] = 0.66;
	}
}

RBM::~RBM()
{
	free(weights);
	free(bh);
	free(bv);
}

__global__ void initialise_curand_on_kernels(curandState * state, unsigned long long seed)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed,row*32 + col, 0, &state[row*blockDim.y * 32 + col]);
}

void RBM::train(int ** samples, int n_samples)
{
	double *devWeights, *devBv, *devBh;
	double *ph_mean, double *nv_means, double *nh_means;
	int *nh_samples, *nv_samples, *ph_sample, *nv_sample;
	nv_sample = (int*)malloc(n_visible * sizeof(int));
	bool *devMask;
	//make device copys
	cudaError_t cudaStatus;
	cudaStatus=cudaMalloc((void **)&devWeights, n_hidden * n_visible * sizeof(double));
	//dummy shared
	cudaStatus = cudaMalloc((void **)&ph_mean, n_hidden  * sizeof(double));
	cudaStatus = cudaMalloc((void **)&ph_sample, n_hidden  * sizeof(int));
	cudaStatus = cudaMalloc((void **)&nv_means,  n_visible * sizeof(double));
	cudaStatus = cudaMalloc((void **)&nv_samples, n_visible * sizeof(int));
	cudaStatus = cudaMalloc((void **)&nh_means, n_hidden * sizeof(double));
	cudaStatus = cudaMalloc((void **)&nh_samples, n_hidden * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda malloc failed!");

	}
	cudaStatus=cudaMalloc((void **)&devBh, n_hidden * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda malloc failed!");

	}
	cudaStatus=cudaMalloc((void **)&devBv, n_visible * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda malloc failed!");

	}
	cudaStatus=cudaMalloc((void **)&devMask, n_hidden * n_visible * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda malloc failed!");

	}

	cudaStatus=cudaMemcpy(devWeights, weights, n_hidden*n_visible * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda memcpy failed!");
		
	}
	cudaStatus=cudaMemcpy(devBv, bv, n_visible * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda memcpy failed!");
		
	}
	cudaStatus=cudaMemcpy(devBh, bh, n_hidden * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda memcpy failed!");
		
	}
	cudaStatus=cudaMemcpy(devMask, mask, n_hidden*n_visible * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda memcpy failed!");
		
	}

	

	int maxNum = std::fmax(n_hidden, n_visible);
	dim3 threads(32, 32);

	dim3 blocks((maxNum + 1) / 32, (maxNum + 1) / 32);
	curandState *globalState;
	unsigned int theSize = 3*32 * 32 * blocks.x * blocks.y * sizeof(curandState);
	printf("globalState size: %d\n", theSize);
	cudaMalloc((void **)&globalState,theSize);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "init curand failed failed! %d\n");
	}
	unsigned long long seed = static_cast<unsigned long long>(time(NULL));
	initialise_curand_on_kernels<<<blocks, threads>>> (globalState,seed);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "init curand failed failed! %d\n");
	}
	for (int j = 0; j < 100; j++) {
		for (int i = 0; i < n_samples; i++) {
			int *sample;
			cudaMalloc((void**)&sample, n_visible * sizeof(int));
			int test = samples[i][n_visible - 1];
			cudaMemcpy(sample, samples[i], n_visible * sizeof(int), cudaMemcpyHostToDevice);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cuda memcpy failed!");
				break;
			}

			contrastive_divergence <<<blocks, threads>>> (globalState,sample, devWeights, devBh, devBv, devMask, ph_mean, ph_sample, nv_means, nv_samples, nh_means, nh_samples, n_hidden, n_visible, 0.08);
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "contrastive failed failed! %d\n", i);
			}

			
			cudaFree(sample);
		}
		printf("epoch (%d) \n", j);
		cudaMemcpy(nv_sample, nv_samples, n_visible * sizeof(int), cudaMemcpyDeviceToHost);
		double mag = 0.0, ene = 0.0;
		for (int i = 0; i < n_visible; i++) {
			int spin2 = nv_sample[i] == 0 ? -1 : 1;
			mag += spin2;
			if (i == 0) {
				int spin1 = nv_sample[n_visible - 1] == 0 ? -1 : 1;
				
				ene += -1.0 *(spin1 * spin2);
			}
			else {
				int spin1 = nv_sample[i-1] == 0 ? -1 : 1;
				int spin2 = nv_sample[i] == 0 ? -1 : 1;
				ene += -1.0 *(spin1 * spin2);
			}
		}
		mag /= n_visible;
		ene /= n_visible;
		printf("mag: %f, ene: %f\n", mag, ene);
	}
	//finished copy back 
	
	
	cudaMemcpy(weights,devWeights,  n_hidden*n_visible * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(bv, devBv,  n_visible * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(bh, devBh,  n_hidden * sizeof(double), cudaMemcpyDeviceToHost);


	
	cudaFree(devWeights);
	cudaFree(devBv);
	cudaFree(devBh);
	cudaFree(devMask);
}

void RBM::printWeights() {
	std::ofstream out("weights.csv");
	int counter = 0;
	for (int i = 0; i < n_visible; i++) {
		for (int j = 0; j < n_hidden; j++) {
			//printf("%f,", weights[i * n_hidden + j]);
			double theVal = weights[i*n_hidden + j];
			if (!isnan(theVal) && abs(theVal) > 0) {
				mean += theVal;
				counter++;
			}
			out << weights[i*n_hidden + j];
			out << ", ";
		}
		out << "\n";
	}
	out.close();
	mean /= counter;
	printf("theMeanJ: %f\n", mean);
}

