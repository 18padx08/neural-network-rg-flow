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

using namespace ct;
int main()
{
	shared_ptr<OptPlaceholder> pl = make_shared<OptPlaceholder>(OptPlaceholder("x"));
	shared_ptr<Variable> var =  make_shared<Variable>();
	var->value = make_shared<Tensor>(Tensor({ 2 }, { 5,-5 }));
	shared_ptr<Sigmoid> sig = make_shared<Sigmoid>(Sigmoid(var));
	shared_ptr<Graph> graph = make_shared<Graph>(Graph(sig));
	Session s(graph);
	map<string, shared_ptr<Tensor>> feedDic = { {"x", make_shared<Tensor>(Tensor({2}, {5,-5}))} };
	s.run(feedDic, true, 0);
	std::cout << (*s.cachedOutput)[{0}] << "," << (*s.cachedOutput)[{1}] << std::endl;
	//std::cout << (double)(*s.cachedOutput);
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
