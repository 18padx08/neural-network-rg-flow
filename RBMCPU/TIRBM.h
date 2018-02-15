#pragma once
#include "Symmetry.h"
#include <vector>
#include <algorithm>
#include <time.h>
#include <ctime>
#include <random>
#include "RBM.h"
using namespace std;
class TIRBM
{
private:
	vector<Symmetry<int>> symmetries;
	//weights
	vector<vector<double>> wij;
	//last changes for the weights (e.g. used for momentum)
	vector<vector<double>> dWij;
	//use for temporary changes to assure thread safty with parallel programming
	vector<vector<double>> tmpDwij;
	//hidden bias matrix
	vector<vector<double>> bjs;
	vector<vector<double>> dbjs;
	//visible biases 
	vector<double> ci;
	vector<double> dci;

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
	void sample_h_given_v(vector<double> &vis_src, vector<double> &hid_target, vector<double> &hid_target_sample);
	//negative step
	void sample_v_given_h(vector<double> &hid_src, vector<double> &vis_target, vector<double> &vis_target_sample);
	//approximate the gradient for the log likelyhood
	double contrastive_divergence(vector<vector<double>> &input, int cdK, int batchSize);
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
	void train(vector<vector<double>> &input, int sample_size, int epoch);
	//sample from the net using gibbs_steps of gibbs sampling iterations
	double * sample_from_net(int gibbs_steps = 10);
	//reconstruct the data using a fixed number of gibbs iterations
	double * reconstruct(vector<double> &input, int gibbs_steps = 1);
	//save the state of the net to a file
	void saveToFile(std::string filename);
	//save a visualization of different parameters for visible check of algorithm
	void saveVisualization();
	//load a previously saved state
	bool loadWeights(std::string filename);
	//helper function for dbms (propagate to next layer)
	void propagate_down(vector<vector<double>> &input, vector<vector<double>> &output, int sample_size);
	//helper function for dbms (propagate to previous layer)
	void propagate_up(vector<vector<double>> &input, vector<vector<double>> &output,  int gibbs_steps = 10);
};

