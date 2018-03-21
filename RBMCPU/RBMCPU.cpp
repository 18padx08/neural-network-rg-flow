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

using namespace ct;
int main()
{
	auto graph = RBMCompTree::getRBMGraph();
	auto session = make_shared<Session>(Session(graph));
	Ising1D ising(500, 0.8, 1);
	vector<int> dims = { 500 };
	vector<double> values(500*50);
	for (int i = 0; i < 20000; i++) {
		ising.monteCarloStep();
	}
	auto conf = ising.getConfiguration();
	for (int i = 0; i < conf.size(); i++) {
		values[i] = conf[i] <=0? -1 : 1;
	}

	map<string, shared_ptr<Tensor>> feedDic = { { "x", make_shared<Tensor>(Tensor(dims,values)) } };
	auto var = dynamic_pointer_cast<Variable>(graph->variables[0]);
	double average = 0;
	int counter = 1;
	auto correlationNN = ising.calcExpectationValue(1);
	auto correlationNNN = ising.calcExpectationValue(2);
	optimizers::ContrastiveDivergence cd(graph,0.1, 0);
	dims = { 500,50 };
	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < 1; j++) {
			session->run(feedDic, true, 1);
			cd.optimize(1,correlationNNN);
		}
		std::cout << "\r" << "                                                                                                         ";
		std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / counter << ")";
		if (i > 800) {
			average += std::abs((double)*(var->value));
			std::cout << "\r" << "                                                                                                         ";
			std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / counter << ")";
			counter++;
		}
		for (int s = 0; s < 50; s++) {
			for (int i = 0; i < 1500; i++) {
				ising.monteCarloStep();
			}
			auto tmpconf = ising.getConfiguration();

			for (int i = 0; i < tmpconf.size(); i++) {
				values[i+s*500] = tmpconf[i] <= 0 ? -1 : 1;
			}
		}
		auto t = make_shared<Tensor>(Tensor(dims, values));
		feedDic = { { "x", t} };
	}
	var->value = make_shared<Tensor>(Tensor({ 1 }, { average / counter }));

	//calculate the expectation values 
	//auto correlationNN = ising.calcExpectationValue(1);
	
	int val = 0;
	std::uniform_int_distribution<int> dist(0, 249);
	auto engine = std::default_random_engine(time(NULL));
	auto castNode = dynamic_pointer_cast<Storage>(graph->storages["hiddens_pooled"]);
	for (int i = 0; i < 5000; i++) {
		auto index = dist(engine);
		session->run(feedDic, true, 5);
		auto size = (*castNode->storage[4]).dimensions[0];
		auto val1 = (*castNode->storage[4])[{index}];
		auto val2 = (*castNode->storage[4])[{index + 1 < size ? index + 1 : 0}];
		val += val1 * val2;

		for (int i = 0; i < 1500; i++) {
			ising.monteCarloStep();
		}
		auto tmpconf = ising.getConfiguration();

		for (int i = 0; i < tmpconf.size(); i++) {
			values[i] = tmpconf[i] <= 0 ? -1 : 1;
		}

		feedDic = { { "x", make_shared<Tensor>(Tensor(dims,values)) } };
	}
	auto corrNNN = (double)val / 5000;

	std::cout << "Monte Carlo measurements vs. NN" << std::endl;
	std::cout << "<v_i v_{i+1}> ~ " << correlationNN << "  exact => " << tanh( 0.8* 1) << " error: " << abs(correlationNN - tanh(0.8)) << std::endl;
	std::cout << "<v_i v_{i+2}> ~ " << correlationNNN << " exact => " << pow(tanh(0.8 * 1), 2) << " error: " << abs(correlationNNN - pow(tanh(0.8),2)) <<  std::endl;
	std::cout << "<h_i h_{i+1}> = " << corrNNN << " error: " << abs(corrNNN - pow(tanh(0.8),2)) <<std::endl;

	
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
