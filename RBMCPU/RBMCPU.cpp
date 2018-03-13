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
#include "Tensor.h"

int main()
{
	ct::Tensor t({ 5,5 });
	t[{0, 0}] = 1;
	t[{1, 0}] = 2;
	t[{0, 1}] = 3;
	t[{1, 1}] = 4;
	t[{4, 4}] = 5;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {	
				std::cout << t[{i, j}] << ",";
		}
		std::cout << std::endl << std::endl;
	}
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
