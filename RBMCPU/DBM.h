#pragma once
#include "RBM.h"
#include <vector>
class DBM
{
	std::vector<RBM> rbms;
	ParamSet parameters;
public:
	DBM(int num_layer, int *layerDimensions, ParamSet parameters, FunctionType activationFunction);
	void train(double **input, int sample_size, int epoch);
	double * sample_from_net(int gibbs_steps = 10);
	double * reconstruct(double * input);
	void setParameters(ParamSet set);
	void printWeights();
	void saveToFile(std::string filename);
	void saveVisualization();
	bool loadWeights(std::string filename);
};

