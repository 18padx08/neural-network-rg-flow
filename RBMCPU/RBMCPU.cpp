// RBMCPU.cpp : Defines the entry point for the console application.

//TODO:
// two spin setup -> check if correct results
// enforce Z2 symmetry
// check on lattice -> compare filters to paper

#pragma once
#include <omp.h>
#include "../CudaTest/Ising1D.h"
#include "MNISTTest.h"
#include "SymmetryTest.h"
#include "TIRBMTest.h"
#include "AnalyticalTest.h"
#include "RG.h"
#include "RBM.h"
#include "targetver.h"
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include "Session.h"
#include "Graph.h"
#include "Add.h"
#include "Sigmoid.h"
#include "RGLayer.h"
#include "ProbPooling.h"
#include "RBMCompTree.h"
#include "ContrastiveDivergence.h"
#include "RGFlowTest.h"
#include "ErrorAnalysis.h"
#include "Phi4Test.h"
#include "FFTTest.h"
#include "TestFFTUpdate.h"
#include "TestLoop.h"

using namespace ct;
int main()
{
	TestLoop loop;
	loop.run();
	//TestFFTUpdate test;
	//test.runFFTCompareToNewHidden();
	/*std::cout << "-- beta = 0.6 --" << std::endl;
	test.plotConvergence(0.6);
	std::cout << "-- beta = 0.8 --" << std::endl;
	test.plotConvergence(0.8);*/
	/*std::cout << "-- beta = 1.0 --" << std::endl;
	test.plotConvergence(1.0);
	std::cout << "-- beta = 1.2 --" << std::endl;
	test.plotConvergence(1.2);*/
	//std::cout << "-- beta = 1.4 --" << std::endl;
	//test.plotConvergence(1.4);
	//std::cout << "--error test--" << std::endl;
	//test.plotError({ 100,1000,10000 });
	//std::cout << "-- beta = 0.8 --" << " (modCD Test)" << std::endl;
	//test.modTest(0.8);
	//test.testGibbsConvergence();
	//test.cheatTest(1.4);
	///test.plotRGFlow(1.0);
	vector<double> couplings = { 0.4,0.3,0.4 };
	vector<double> bs = { 40,20,40,80 };
	/*for (auto c : couplings) {
		for (int i = 10; i <= 80; i *= 2) {
			std::cout << std::endl <<  "-- bs=" << i << " --" << std::endl;
			RGFlowTest test;
			test.plotRGFlowNew(c, i);
		}
	}*/
	//Phi4Test test;
	//test.run();
	/*for (int i = 0; i < couplings.size(); i++) {
		for (int j = 0; j < bs.size(); j++) {
			std::cout << std::endl << "-- bs=" << i << " --" << std::endl;
			ErrorAnalysis analysis;
			analysis.plotErrorOnTraining(couplings[i], bs[j]);
		}
	}*/

	/*srand(time(NULL));
	TIRBMTest tTest;
	tTest.runTest();
	//tTest.runMnist();
	MNISTTest test;
	//test.executeRBMCPU();
	int chainLength = 10;
	*/
	/*int steps = 5000;
	vector<double> couplings = { 0.1,0.2,0.5,1 };
	std::vector<vector<double>> magnetization(5, std::vector<double>(steps));
	std::vector<vector<double>> th(5, std::vector<double>(steps));
	for (int mag = 0; mag < 5; mag++) {
		Ising1D ising(chainLength, couplings[mag], 1);
		for (int step = 0; step < steps; step++) {
			ising.monteCarloStep();
			magnetization[mag][step] = ising.getMagnetization();
			th[mag][step] = 0;// ising.getTheoreticalMeanEnergy();
		}
	}
	std::ofstream output("ising_output.csv");
	for (int step = 0; step < magnetization[0].size(); step++) {
		output << step << ",";
		for (int mag = 0; mag < 5; mag++) {
			output << magnetization[mag][step] << "," << th[mag][step] << ",";
		}
		output << std::endl;
	}
	output.close();*/
	return 0;
}
