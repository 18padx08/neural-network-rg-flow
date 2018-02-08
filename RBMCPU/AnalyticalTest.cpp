#include "AnalyticalTest.h"
#include "RBM.h"
#include <iostream>
#include <stdio.h>

void AnalyticalTest::runTest()
{
	ParamSet set;
	set.lr = 0.1;
	set.momentum = 0.3;
	set.regulization = (Regularization)(Regularization::DROPCONNECT| Regularization::L1) ;
	RBM rbm(2, 2, FunctionType::SIGMOID);
	rbm.setParameters(set);
	bool **mask = (bool **)malloc(2*sizeof(bool*));
	mask[0] = (bool *)malloc(2);
	mask[0][0] = true;
	mask[0][1] = false;
	mask[1] = (bool*) malloc(2);
	mask[1][0] = false;
	mask[1][1] = true;
	rbm.initMask(mask);
	//rbm.loadWeights("thanalyticaltest.csv");
	double v1[] = { 0,0 };
	double v2[] = { 0,1 };
	double v3[] = { 1,0 };
	double v4[] = { 1,1 };
	//rbm.initMask();
	rbm.initWeights();
	double **samples = (double **)malloc(sizeof(double *) * 4);
	for (int i = 0; i < 4; i++) {
		samples[i] = (double *)malloc(sizeof(double) * 2);
		if(i==0)
			samples[i] = v1;
		if (i == 1)
			samples[i] = v2;
		if (i == 2)
			samples[i] = v3;
		if (i == 3)
			samples[i] = v4;
	}
	rbm.train(samples, 4, 5000);
	rbm.saveToFile("analyticaltest.csv");
	double *r1 = rbm.reconstruct(v1);
	double *r2 = rbm.reconstruct(v2);
	double *r3 = rbm.reconstruct(v3);
	double *r4 = rbm.reconstruct(v4);
	std::cout << std::endl;
	std::cout << v1[0] << v1[1] << std::endl;
	std::cout << r1[0] << r1[1] << std::endl;
	std::cout << "---------" << std::endl;
	std::cout << v2[0] << v2[1] << std::endl;
	std::cout << r2[0] << r2[1] << std::endl;
	std::cout << "---------" << std::endl;
	std::cout << v3[0] << v3[1] << std::endl;
	std::cout << r3[0] << r3[1] << std::endl;
	std::cout << "---------" << std::endl;
	std::cout << v4[0] << v4[1] << std::endl;
	std::cout << r4[0] << r4[1] << std::endl;
	std::cout << "---------" << std::endl;
 }

void AnalyticalTest::runAnalytical()
{
	RBM rbm(2, 2, FunctionType::SIGMOID);
	
	double v1[] = { 0,0 };
	double v2[] = { 0,1 };
	double v3[] = { 1,0 };
	double v4[] = { 1,1 };
	rbm.loadWeights("thanalyticaltest.csv");
	double *r1 = rbm.reconstruct(v1);
	double *r2 = rbm.reconstruct(v2);
	double *r3 = rbm.reconstruct(v3);
	double *r4 = rbm.reconstruct(v4);
	std::cout << std::endl;
	std::cout << v1[0] << v1[1] << std::endl;
	std::cout << r1[0] << r1[1] << std::endl;
	std::cout << "---------" << std::endl;
	std::cout << v2[0] << v2[1] << std::endl;
	std::cout << r2[0] << r2[1] << std::endl;
	std::cout << "---------" << std::endl;
	std::cout << v3[0] << v3[1] << std::endl;
	std::cout << r3[0] << r3[1] << std::endl;
	std::cout << "---------" << std::endl;
	std::cout << v4[0] << v4[1] << std::endl;
	std::cout << r4[0] << r4[1] << std::endl;
	std::cout << "---------" << std::endl;
}
