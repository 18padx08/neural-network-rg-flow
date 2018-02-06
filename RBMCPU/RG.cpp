#include "RG.h"
#include "RBM.h"
#include <stdio.h>
#include "../CudaTest/Ising1D.h"
#include "SymmetryCombination.cpp"
#include "TranslationSymmetry.cpp"
#include <fstream>
#include <iostream>
#include <thread>


RG::RG()
{
}


RG::~RG()
{
}

//make a 20 spin chain and try to learn
void RG::runRG()
{
	int sampleSize = 100;
	double theoreticalEnergy = 0;
	double **samples = (double **)malloc(sampleSize * sizeof(double*));
	double **tmpSamples = (double **)malloc(sampleSize * sizeof(double*));
	for (int i = 0; i < sampleSize; i++) {
		Ising1D ising(20, 1, 0.7);
		int counter = 0;
		double mE, tE, M;
		do {
			ising.monteCarloStep();
			counter++;
			mE = ising.getMeanEnergy();
			tE = ising.getTheoreticalMeanEnergy();
			theoreticalEnergy = tE;
			M = ising.getMagnetization();

		} while (!(abs(tE - mE) < 0.1 * abs(tE) && abs(ising.getMagnetization()) < 0.01));

		if (i % 2 == 0) {
			//printf("[STEP %d] Mean energy config: %f theoretical: %f delta: %f\n Mean magnetization: %f\n", i, ising.getMeanEnergy(), ising.getTheoreticalMeanEnergy(), ising.getMeanEnergy() - ising.getTheoreticalMeanEnergy(), ising.getMagnetization());
		}
		samples[i] = (double *)malloc(20 * sizeof(double));
		tmpSamples[i] = (double *)malloc(20 * sizeof(double));
		std::vector<int> v = ising.getConfiguration();
		for (int j = 0; j < v.size(); j++) {
			samples[i][j] = v[j];
		}
	}


	bool **mask = (bool**)malloc(10 * 20 * sizeof(bool));
	int maskCounter = 0;
	int lastRow = 2;
	int lastCol = 0;
	bool second = false;
	for (int i = 0; i < 20; i++) {
		mask[i] = (bool*)malloc(10 * sizeof(bool));
	}
	mask[0][0] = true;
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 10; j++) {
			if (i % 2 == 0)
			{

				if (i == lastRow && j == lastCol) {
					mask[i][j] = true;
					maskCounter++;
					if (!second) {
						lastCol += 1;
						second = true;
					}
				}
				else {
					if (i == 0 && j == 0) {
						mask[0][0] = true;
					}
					else {
						mask[i][j] = false;
					}
				}
			}
			else {
				mask[i][j] = false;
			}

		}
		if (i % 2 == 0 && i > 0) {
			second = false;
			lastRow += 2;
		}
	}
	RBM rbm(20, 10, FunctionType::SIGMOID);
	rbm.loadWeights("weights_ising.csv");
	ParamSet set;
	set.lr = 0.01;
	set.momentum = 0.3;
	set.regulization = (Regularization)( Regularization::DROPCONNECT);
	rbm.setParameters(set);
	rbm.initMask(mask);
	rbm.initWeights();
	TranslationSymmetry<double> *t = new TranslationSymmetry<double>();
	long timeStart = time(NULL);
	//permute once through the chain
	for (int i = 0; i < 2; i++) {
		for (int trans = 0; trans < 5; trans++) {
			long loopStart = time(NULL);
			for (int ba = 0; ba < sampleSize; ba++) {
				(*t)(samples[ba], tmpSamples[ba], 20);
			}
			rbm.train(tmpSamples, sampleSize, 50);
			rbm.saveToFile("weights_ising.csv");
			std::cout << std::endl;
			long deltaT = time(NULL) - loopStart;
			long total = time(NULL) - timeStart;
			long estimated = (20 - trans) * deltaT;
			std::cout << "Time elapsed: " << total << "s of estimated " << estimated / 60 << "min " << estimated % 60 << "s" << std::endl;
		}
	}
	
	//rbm.saveToFile("weights_ising.csv");
	long timeEnd = time(NULL);
	//rbm.saveVisualization();
	double totalMagn = 0;
	double totalEnergy = 0;
	for (int trials = 0; trials < 100; trials++) {
		double *sample;
		sample = rbm.sample_from_net();
		double magn = 0;
		double energy = 0;
		std::cout << " --------- " <<std::endl;
		for (int i = 0; i < 20; i++) {
			std::cout << samples[0][i];
		}
		std::cout << std::endl;
		for (int i = 0; i < 20; i++) {
			
			std::cout << sample[i];
			magn += sample[i] <= 0? -1 : 1;
		}
		std::cout << std::endl;
		magn /= 20.0;
		for (int i = 0; i < 19; i++) {
			energy += -1 * (sample[i] <= 0? -1 :1) * (sample[i + 1] <= 0? -1 : 1);
		}
		energy += -1 * (sample[19] <= 0 ? -1 : 1) * (sample[0] <= 0? -1 : 1);
		energy /= 20.0;
		totalEnergy += energy;
		totalMagn += magn;
		std::cout << energy << "  " << magn << std::endl;
	}
	totalMagn /= 100;
	totalEnergy /= 100;


	std::cout << "Trained for " << timeEnd - timeStart << "s" << std::endl;
	std::cout << std::endl << "--Theoretical--" << std::endl;
	std::cout << "Energy: " << theoreticalEnergy << "| Magnetization: " << 0 << std::endl;
	std::cout << "--Network prediction--" << std::endl;
	std::cout << "Energy: " << totalEnergy << " | Magnetization: " << totalMagn << std::endl;

	/*std::cout << "----- Load theoretical weights -----" << std::endl;
	RBM rbm2(20, 10);
	rbm2.loadWeights("theoretical_test.csv");
	totalMagn = 0;
	totalEnergy = 0;
	for (int trials = 0; trials < 20; trials++) {
		double *sample;
		sample = rbm2.sample_from_net();
		double magn = 0;
		double energy = 0;
		for (int i = 0; i < 20; i++) {
			std::cout << sample[i];
			magn += sample[i] <= 0 ? -1 : 1;
		}
		std::cout << std::endl;
		magn /= 20.0;
		for (int i = 0; i < 19; i++) {
			energy += -1 * (sample[i] <= 0 ? -1 : 1) * (sample[i + 1] <= 0 ? -1 : 1);
		}
		energy += -1 * (sample[19] <= 0 ? -1 : 1) * (sample[0] <= 0 ? -1 : 1);
		energy /= 20.0;
		totalEnergy += energy;
		totalMagn += magn;
		std::cout << energy << "  " << magn << std::endl;
	}
	totalMagn /= 20;
	totalEnergy /= 20;

	std::cout << std::endl << "--Theoretical--" << std::endl;
	std::cout << "Energy: " << theoreticalEnergy << "| Magnetization: " << 0 << std::endl;
	std::cout << "--Network prediction--" << std::endl;
	std::cout << "Energy: " << totalEnergy << " | Magnetization: " << totalMagn << std::endl;*/
}
