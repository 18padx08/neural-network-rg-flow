#include "MNISTTest.h"
#include "MNISTData.h"
#include "RBM.h"
#include "DBM.h"
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
	catch (std::runtime_error ex) {
		std::cout << "Catched runtime_error [Passed]" << std::endl;
	}
	try {
		RBM rbm(30 * 30, 30 * 30 / 2);
		rbm.loadWeights("weights_with_reg.csv");
		std::cout << "No error thrown [FAILED]" << std::endl;
	}
	catch (std::runtime_error ex) {
		std::cout << "Catched runtime_error [Passed]" << std::endl;
	}
	std::cout << " Test finished" << std::endl;


}
void MNISTTest::intenseTest()
{

	RBM rbm(28 * 28, 28 * 28 / 2, FunctionType::SIGMOID);
	//rbm.initMask();
	//rbm.initWeights();
	ParamSet set;
	set.lr = 0.1;
	set.momentum = 0.6;
	set.regulization = (Regularization)( Regularization::L1);
	
	//rbm.setParameters(set);
	long long starttime = time(NULL);
	MNISTData data;
	/*for (int i = 0; i < 1000; i++) {
		double **batch = data.getBatch(50);
		rbm.train(batch, 50, 10);
		//save weights to file
		std::cout << "saving files to weights_with_reg_all_numbers.csv" << std::endl;
		rbm.saveToFile("weights_with_reg_all_numbers.csv");
	}*/
	rbm.loadWeights("weights_with_reg_all_numbers.csv");
	std::cout << "training finished in " << time(NULL) - starttime << "s" << std::endl;
	//save visualization
	rbm.saveVisualization();
	double **batch = data.getBatch(20);
	for (int sam = 0; sam < 20; sam++) {
		double * sample = rbm.reconstruct(batch[sam]);
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				std::cout << sample[i * 28 + j];
			}
			std::cout << std::endl;
		}
	}/*
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
}
void MNISTTest::DBNTest()
{
	int dimensions[] = {28*28,28*28/2,28*28/4,28*28/8,28*28/16};
	ParamSet set;
	set.lr = 0.1;
	set.momentum = 0.5;
	set.regulization = Regularization::L1;
	set.weightDecay = 2e-4;
	DBM dbm(4,dimensions, set, FunctionType::SIGMOID);
	MNISTData data;
	for (int i = 0; i < 100; i++) {
		double **batch = data.getBatch(50);
		dbm.train(batch, 50, 10);
		dbm.saveToFile("dbm_mnist");
	}
	dbm.saveVisualization();
	for (int o = 0; o < 100; o++) {
		double * sample = dbm.sample_from_net(100);
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				std::cout << sample[i * 28 + j];
			}
			std::cout << std::endl;
		}
		std::cout << std::endl << std::endl;
	}
}
void MNISTTest::executeRBMCPU()
{
	RBM rbm(28*28, 500);
	rbm.initWeights();
	MNISTData data;
	for (int i = 0; i < 100; i++) {
		double **batch = data.getBatch(10);
		
		rbm.train(batch, 10, 5);
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
	set.regulization = Regularization::NONE;
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
