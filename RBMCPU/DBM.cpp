#include "DBM.h"
#include <stdio.h>
#include <iostream>
#include <assert.h>

DBM::DBM(int num_layer, int *layerDimensions, ParamSet parameters, FunctionType activationFunction)
{
	for (int i = 0; i < num_layer; i++) {
		RBM rbm(layerDimensions[i], layerDimensions[i + 1], activationFunction);
		rbm.setParameters(parameters);
		rbm.initWeights();
		rbms.push_back(rbm);
	}
}

void DBM::train(double ** input, int sample_size, int epoch)
{
	//iteratively train networks
#pragma omp parallel for
	for (int r = 0; r < rbms.size(); r++) {
		if (r == 0) {
			rbms[r].train(input, sample_size, epoch);
			continue;
		}
		double **tmpInput = nullptr;
		for (int prop = 0; prop < r; prop++) {
			 tmpInput = rbms[prop].propagate(input, sample_size);
		}
		rbms[r].train(tmpInput, sample_size, epoch);		
		delete(tmpInput);
	}
}

double * DBM::sample_from_net(int gibbs_steps)
{
	double *last;
	last = rbms[rbms.size()-1].sample_from_net(gibbs_steps);
	for (int i = rbms.size() - 2; i >= 0; i--) {
		last = rbms[i].propup(last, gibbs_steps);
	}
	return last;
}

double * DBM::reconstruct(double * input)
{
	return nullptr;
}

void DBM::setParameters(ParamSet set)
{
}

void DBM::printWeights()
{
}

void DBM::saveToFile(std::string filename)
{
	for (int i = 0; i < rbms.size(); i++) {
		rbms[i].saveToFile(filename +"_layer_" + std::to_string(i)+".csv");
	}
}

void DBM::saveVisualization()
{
	rbms[0].saveVisualization();
}

bool DBM::loadWeights(std::string filename)
{
	for (int i = 0; i < rbms.size(); i++) {
		rbms[i].loadWeights(filename + "_layer_" + std::to_string(i) + ".csv");
	}
	return true;
}

bool DBM::initMask(bool ** mask)
{
	for (int i = 0; i < rbms.size(); i++) {
		rbms[i].initMask(mask);
		rbms[i].initWeights();
	}

	return true;
}

void DBM::startAveraging()
{
	for (int i = 0; i < rbms.size(); i++) {
		rbms[i].startAveraging();
	}

}
