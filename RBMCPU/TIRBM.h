#pragma once
#include "Symmetry.h"
#include <vector>
#include <algorithm>
#include <time.h>
#include <ctime>
#include <random>
#include "RBM.h"
class TIRBM
{
private:
	std::vector<Symmetry<int>> symmetries;
	//weights
	double **wij;
	//last changes for the weights (e.g. used for momentum)
	double **dWij;
	//use for temporary changes to assure thread safty with parallel programming
	double **tmpDwij;
	//hidden bias matrix
	double **bjs;
	double **dbjs;
	//visible biases 
	double *ci;
	double *dci;

	//parameters used for learning
	ParamSet parameters;
	//Random number generator
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;

	//sample from bernoulli dist
	double bernoulli(double p);
	//sample from unifrom dist
	double uniform(double min, double max);
	//positive step
	void sample_h_given_v(double *vis_src, double *hid_target, double *hid_target_sample);
	//negative step
	void sample_v_given_h(double *hid_src, double *vis_target, double *vis_target_sample);
	//approximate the gradient for the log likelyhood
	double contrastive_divergence(double **input, int cdK, int batchSize);
	ActivationFunction actFun;

public:
	TIRBM(int n_vis, int n_hid) : TIRBM(n_vis, n_hid, FunctionType::SIGMOID) {};
	TIRBM(int n_vis, int n_hid, FunctionType activationFunction);
	~TIRBM();
	//initialize the weights (ensure to set them to some value) (default double is 3e66)
	void initWeights();
	//set the parameters for the learning algorithm such as learning rate, momentum, activationFunction, regularization
	void setParameters(ParamSet set);
	//train for number of epochs with a test set
	void train(double **input, int sample_size, int epoch);
	//sample from the net using gibbs_steps of gibbs sampling iterations
	double * sample_from_net(int gibbs_steps = 10);
	//reconstruct the data using a fixed number of gibbs iterations
	double * reconstruct(double * input, int gibbs_steps = 1);
	//save the state of the net to a file
	void saveToFile(std::string filename);
	//save a visualization of different parameters for visible check of algorithm
	void saveVisualization();
	//load a previously saved state
	bool loadWeights(std::string filename);
	//helper function for dbms (propagate to next layer)
	void propagate_down(double **input,double **output, int sample_size);
	//helper function for dbms (propagate to previous layer)
	void propagate_up(double **input, double **output,  int gibbs_steps = 10);
};

