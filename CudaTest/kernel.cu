
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "RBM.cuh"
#include "Ising1D.h"
#include <fstream>
#include <iostream>
#include <thread>



__host__
void generateSpins(int **samples);
__host__
void readSpins(int **samples);
__host__
void theIsing(int **samples);

#define NUM_THREADS 4

int main()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	srand(time(NULL));
	int **samples = (int **)malloc(1000 * sizeof(int*));

	generateSpins(samples);
	
	
	//why 3x3?
	bool *mask = (bool*)malloc(500 * 1000* sizeof(bool));
	int maskCounter = 0;
	int lastRow = 2;
	int lastCol = 0;
	bool second = false;
	mask[0] = true;
	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < 500; j++) {
			if (i % 2 == 0)
			{
				
				if (i==lastRow && j == lastCol) {
					mask[i * 500 + j] = true;
					maskCounter++;
					if (!second) {
						lastCol += 1;
						second = true;
					}
				}
				else {
					if (i == 0 && j == 0) {
						mask[0] = true;
					}
					else {
						mask[i * 500 + j] = false;
					}
				}
			}
			else {
				mask[i * 500 + j] = false;
			}
		
		}
		if (i % 2 == 0 && i > 0) {
			second = false;
			lastRow += 2;
		}
	}
	printf("%d values masked\n", maskCounter);

	

	RBM rbm(1000,500,mask);
	rbm.train(samples, 40);
	rbm.printWeights();
	
    return 0;
}

void theIsing(int **samples) {
	for (int i = 0; i < 10; i++) {
		Ising1D ising(1000, 1, 1);
		int counter = 0;
		double mE, tE, M;
		do {
			ising.monteCarloStep();
			counter++;
			mE = ising.getMeanEnergy();
			tE = ising.getTheoreticalMeanEnergy();
			M = ising.getMagnetization();

		} while (!(abs(tE - mE) < 0.1 * abs(tE) && abs(ising.getMagnetization()) < 0.01));

		if (i % 2 == 0) {
			printf("[STEP %d] Mean energy config: %f theoretical: %f delta: %f\n Mean magnetization: %f\n", i, ising.getMeanEnergy(), ising.getTheoreticalMeanEnergy(), ising.getMeanEnergy() - ising.getTheoreticalMeanEnergy(), ising.getMagnetization());
		}
		samples[i] = (int *)malloc(1000 * sizeof(int *));

		std::vector<int> v = ising.getConfiguration();
		std::copy(v.begin(), v.end(), samples[i]);
	}
}
void generateSpins(int **samples) {
	std::thread t1(theIsing, samples);
	std::thread t2(theIsing, samples + 10);
	std::thread t3(theIsing, samples + 20);
	std::thread t4(theIsing, samples + 30);
	
	t1.join();
	t2.join();
	t3.join();
	t4.join();
	

	std::ofstream spinsOut("spins.csv");

	for (int i = 0; i < 40; i++) {
		for (int j = 0; j < 1000; j++) {
			spinsOut << samples[i][j] << ", ";
		}
		spinsOut << std::endl;
	}
	spinsOut.close();
}

void readSpins(int **samples) {
	char yorno = scanf("Generate new samples?[y/N] %c");
	if (yorno == 'y') {
		generateSpins(samples);
		return;
	}
	std::ifstream input("spins.csv");
	//while(line = input.read)
}
