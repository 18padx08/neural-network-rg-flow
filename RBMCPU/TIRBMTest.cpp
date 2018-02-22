#include "TIRBMTest.h"
#include "TIRBM.h"
#include "SymmetryCombination.cpp"
#include "TranslationSymmetry.cpp"
#include "Z2.cpp"
#include "../CudaTest/Ising1D.h"


TIRBMTest::TIRBMTest()
{
}


TIRBMTest::~TIRBMTest()
{
}

void runIsing(double J, int sampleSize, vector<vector<double>> &samples, double *theoreticalEnergy, double *firstEnergy, bool firstTime) {
	std::cout << "Start Ising simulation with: " << "beta(1), " << "J(" << J << "), " << "sampleSize(" << sampleSize << ")" << std::endl;
#pragma omp parallel for
	for (int i = 0; i < sampleSize; i++) {
		Ising1D ising(20, 1, J);
		int counter = 0;
		double mE, tE, M;
		do {
			ising.monteCarloStep();
			counter++;
			mE = ising.getMeanEnergy();
			tE = ising.getTheoreticalMeanEnergy();
			*theoreticalEnergy = tE;
			M = ising.getMagnetization();
		} while (!(abs(tE - mE) < 0.1 * abs(tE) && abs(ising.getMagnetization()) < 0.05));
		if (i == 0) *firstEnergy = mE;
		if (i % 2 == 0) {
			//printf("[STEP %d] Mean energy config: %f theoretical: %f delta: %f\n Mean magnetization: %f\n", i, ising.getMeanEnergy(), ising.getTheoreticalMeanEnergy(), ising.getMeanEnergy() - ising.getTheoreticalMeanEnergy(), ising.getMagnetization());
		}
		
		std::vector<int> v = ising.getConfiguration();
		for (int j = 0; j < v.size(); j++) {
			samples[i][j] = v[j];
		}



	}
}

void TIRBMTest::runTest()
{
	
	int sampleSize = 10;
	double J = 1;
	double theoreticalEnergy = 0;
	vector<vector<double>> samples(sampleSize, std::vector<double>(20));
	double firstEnergy = 0;
	runIsing(J, sampleSize, samples, &theoreticalEnergy, &firstEnergy, true);
	Z2<double> z2;
	vector<Symmetry<double>*> symmetries;
	symmetries.push_back(&z2);

	for (int i = 0; i < 10; i++) {
		TranslationSymmetry<double> *t = new TranslationSymmetry<double>(i);
		symmetries.push_back(t);
	}

	TIRBM tirbm(20, 10, FunctionType::SIGMOID);

	ParamSet set;
	set.lr = 0.07;
	set.momentum = 0.5;
	set.regulization = (Regularization)(Regularization::L1);
	set.weightDecay = 2e-3;
	tirbm.setParameters(set);
	tirbm.setSymmetries(symmetries);
	std::cout << "--Start Training --" << std::endl;
	for (int i = 0; i < 400; i++) {
		tirbm.train(samples, sampleSize, 20);
		runIsing(J, sampleSize, samples, &theoreticalEnergy, &firstEnergy, true);
		std::cout << "Starting next iteration: " << i << std::endl;
	}
	for (int i = 0; i < sampleSize; i++) {
		for (int j = 0; j < 20; j++) {
			std::cout << samples[i][j];
		}
		std::cout << std::endl;
	}

	tirbm.saveToFile("tirbm");
}
