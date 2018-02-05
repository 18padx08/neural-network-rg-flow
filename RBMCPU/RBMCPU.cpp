// RBMCPU.cpp : Defines the entry point for the console application.

#include "MNISTTest.h"
#include "SymmetryTest.h"
#include "RG.h"
#include "RBM.h"
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <ctime>

int main()
{
	srand(time(NULL));

	RG rg;
	rg.runRG();

	///TESTs:
	/*std::cout << "---- MNISTTests ----" << std::endl;
	MNISTTest test;
	test.intenseTest();
	
	//test.loadWeightsRBMCPU();
	std::cout << "---- SymmetryTest ----" << std::endl;
	SymmetryTest st;
	//st.runSymmetryTest();
	test.loadWeightsRBMCPU();
	
	/*RBM rbm(28 * 28, 28 * 28 / 2);
	rbm.loadWeights("weights_with_reg.csv");
	double * sample = rbm.sample_from_net();
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			std::cout << sample[i * 28 + j];
		}
		std::cout << std::endl;
	}
	sample = rbm.sample_from_net();
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			std::cout << sample[i * 28 + j];
		}
		std::cout << std::endl;
	}
	sample = rbm.sample_from_net();
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			std::cout << sample[i * 28 + j];
		}
		std::cout << std::endl;
	}
	sample = rbm.sample_from_net();
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			std::cout << sample[i * 28 + j];
		}
		std::cout << std::endl;
	}
	sample = rbm.sample_from_net();
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			std::cout << sample[i * 28 + j];
		}
		std::cout << std::endl;
	}*/
	
    return 0;
}

