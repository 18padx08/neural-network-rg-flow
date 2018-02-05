#include "MNISTTest.h"
#include "MNISTData.h"
#include "RBM.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <time.h>

MNISTTest::MNISTTest()
{

}
void MNISTTest::loadWeightsRBMCPU() {
	//should work
	std::cout << "loading working weights";
	RBM rbm(28 * 28, 28 * 28 / 2);
	rbm.loadWeights("weights_with_reg.csv");
	std::cout << "  [PASSED]" <<std::endl;
	//dimension too small
	try {
		RBM rbm(20 * 20, 28 * 28 / 2);
		rbm.loadWeights("weights_with_reg.csv");
		std::cout << "No error thrown [FAILED]" << std::endl;
	}
	catch (std::exception ex) {
		std::cout << "Catched exception [Passed]" << std::endl;
	}
	try {
		RBM rbm(30 * 30, 30 * 30 / 2);
		rbm.loadWeights("weights_with_reg.csv");
		std::cout << "No error thrown [FAILED]" << std::endl;
	}
	catch (std::exception ex) {
		std::cout << "Catched exception [Passed]" << std::endl;
	}
	std::cout << " Test finished" << std::endl;


}
void MNISTTest::intenseTest()
{

	RBM rbm(28 * 28, 28 * 28 / 2, FunctionType::SIGMOID);
	rbm.initWeights();
	ParamSet set;
	set.lr = 0.001;
	set.momentum = 0.2;
	set.regulization = (Regulization)( Regulization::L1);
	rbm.setParameters(set);
	long long starttime = time(NULL);
	MNISTData data;
	for (int i = 0; i < 60; i++) {
		double **batch = data.getBatch(1000);
		rbm.train(batch, 1000, 10);
		//save weights to file
		std::cout << "saving files to weights_with_reg.csv" << std::endl;
		rbm.saveToFile("weights_with_reg.csv");
	}
	std::cout << "training finished in " << time(NULL) - starttime << "s" << std::endl;
	//save visualization
	//rbm.saveVisualization();
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
	}
}
void MNISTTest::executeRBMCPU()
{
	RBM rbm(28*28, 28*28/2);
	rbm.initWeights();
	MNISTData data;
	for (int i = 0; i < 5; i++) {
		double **batch = data.getBatch(50);
		
		rbm.train(batch, 50, 10);
		//save weights to file
		std::cout << "saving files to weights_with_reg.csv" << std::endl;
		rbm.saveToFile("weights_with_reg.csv");
	}
	//save visualization
	rbm.saveVisualization();
	//print some reconstructed numbers
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
	}
	std::cout << "Starting training without regularizer" << std::endl;
	rbm.initWeights();
	ParamSet set;
	set.regulization = Regulization::NONE;
	set.lr = 0.01;
	set.momentum = 0.3;
	rbm.setParameters(set);
	for (int i = 0; i < 5; i++) {
		double **batch = data.getBatch(50);
		rbm.train(batch, 50, 10);
		//save weights to file
		std::cout << "saving files to weights_without_reg.csv" << std::endl;
		rbm.saveToFile("weights_without_reg.csv");
	}

	//print some reconstructed numbers
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
	}
	sample = rbm.sample_from_net();
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			std::cout << sample[i * 28 + j];
		}
		std::cout << std::endl;
	}
	std::cout << "Test passed without errors" << std::endl;
}
