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
	int sampleSize = 1500;
	double theoreticalEnergy = 0;
	double **samples = (double **)malloc(sampleSize * sizeof(double*));
	double **tmpSamples = (double **)malloc(sampleSize * sizeof(double*));
	for (int i = 0; i < sampleSize; i++) {
		Ising1D ising(20, 1, 1);
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
	RBM rbm(20, 10);
	ParamSet set;
	set.lr = 0.01;
	set.momentum = 0.2;
	set.regulization = (Regulization) (Regulization::DROPCONNECT | Regulization::L1);
	rbm.setParameters(set);
	rbm.initMask();
	rbm.initWeights();
	TranslationSymmetry<double> *t = new TranslationSymmetry<double>();
	long timeStart = time(NULL);
	for (int i = 0; i < 1; i++) {
		//permute once through the chain
		for (int trans = 0; trans < 20; trans++) {
			long loopStart = time(NULL);
			for (int ba = 0; ba < sampleSize; ba++) {
				(*t)(samples[ba], tmpSamples[ba], 20);
			}
			rbm.train(tmpSamples, sampleSize, 40);
			rbm.saveToFile("weights_ising.csv");
			std::cout << std::endl;
			long deltaT = time(NULL) - loopStart;
			long total = time(NULL) - timeStart;
			long estimated = (20-trans) * deltaT;
			std::cout << "Time elapsed: " << total << "s of estimated " << estimated / 60 << "min " << estimated % 60 << "s" << std::endl;
		}
			}
	long timeEnd = time(NULL);
	double *sample;
	sample = rbm.sample_from_net();

	double magn = 0;
	double energy = 0;
	for (int i = 0; i < 20; i++) {
		magn += sample[i];
	}
	magn /= 20.0;
	for (int i = 0; i < 19; i++) {
		energy += -1 * sample[i] * sample[i + 1];
	}
	energy += -1 * sample[19] * sample[0];
	energy /= 20.0;

	rbm.saveToFile("weights_ising.csv");
	rbm.saveVisualization();

	std::cout << "Trained for " << timeEnd - timeStart << "s" << std::endl;
	std::cout << std::endl << "--Theoretical--" << std::endl;
	std::cout << "Energy: " << theoreticalEnergy << "| Magnetization: " << 0 << std::endl;
	std::cout << "--Network prediction--" << std::endl;
	std::cout << "Energy: " << energy << " | Magnetization: " << magn << std::endl;
}
