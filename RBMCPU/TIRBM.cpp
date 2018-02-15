#include "TIRBM.h"



TIRBM::~TIRBM()
{

}

void TIRBM::initWeights()
{
}

void TIRBM::setParameters(ParamSet set)
{
}

void TIRBM::train(vector<vector<double>>& input, int sample_size, int epoch)
{
}



double * TIRBM::sample_from_net(int gibbs_steps)
{
	return nullptr;
}

double * TIRBM::reconstruct(vector<double>& input, int gibbs_steps)
{
	return nullptr;
}



void TIRBM::saveToFile(std::string filename)
{
}

void TIRBM::saveVisualization()
{
}

bool TIRBM::loadWeights(std::string filename)
{
	return false;
}

void TIRBM::propagate_down(vector<vector<double>>& input, vector<vector<double>>& output, int sample_size)
{
}

void TIRBM::propagate_up(vector<vector<double>>& input, vector<vector<double>>& output, int gibbs_steps)
{
}


double TIRBM::bernoulli(double p)
{
	return 0.0;
}

double TIRBM::uniform(double min, double max)
{
	return 0.0;
}

void TIRBM::sample_h_given_v(vector<double>& vis_src, vector<double>& hid_target, vector<double>& hid_target_sample)
{
}

void TIRBM::sample_v_given_h(vector<double>& hid_src, vector<double>& vis_target, vector<double>& vis_target_sample)
{
}

double TIRBM::contrastive_divergence(vector<vector<double>>& input, int cdK, int batchSize)
{
	return 0.0;
}



TIRBM::TIRBM(int n_vis, int n_hid, FunctionType activationFunction) : generator(time(NULL)), distribution(0.0,1.0), actFun(activationFunction)
{
	//allocate memory for all arrays
	//we need one hidden unit bias per symmetry generator

}
