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
	double J = 0.7;
	double theoreticalEnergy = 0;
	vector<vector<double>> samples(sampleSize, std::vector<double>(20));
	double firstEnergy = 0;
	runIsing(J, sampleSize, samples, &theoreticalEnergy, &firstEnergy, true);
	TranslationSymmetry<double> t1(1);
	TranslationSymmetry<double> t2(2);
	TranslationSymmetry<double> t3(4);
	Z2<double> z2;
	vector<double> test = { 0,1,2,3,4 };
	auto test2 = t1(test);
	vector<Symmetry<double>*> symmetries;
	symmetries.push_back(&t1);
	symmetries.push_back(&t2);
	symmetries.push_back(&t3);
	//symmetries.push_back(&z2);

	TIRBM tirbm(10, 5, FunctionType::SIGMOID);
	
	ParamSet set;
	set.lr = 0.1;
	set.momentum = 0.0;
	set.regulization = Regularization::L1;
	set.weightDecay = 2e-4;
	tirbm.setParameters(set);
	tirbm.setSymmetries(symmetries);
	std::cout << "--Start Training --" << std::endl;
	for (int i = 0; i < 200; i++) {
		tirbm.train(samples, sampleSize, 10);
		runIsing(J, sampleSize, samples, &theoreticalEnergy, &firstEnergy, true);
		std::cout << "Starting next iteration: " << i << std::endl;
	}

	tirbm.saveToFile("tirbm");
}
