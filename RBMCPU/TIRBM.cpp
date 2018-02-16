#include "TIRBM.h"
#include <iostream>
#include <fstream>


TIRBM::~TIRBM()
{

}

void TIRBM::setSymmetries(vector<Symmetry<double> *> &symmetries)
{
	n_sym = symmetries.size();
	this->bjs = vector<vector<double>>(n_hid, vector<double>(n_sym));
	for (auto sym : symmetries) {
		this->symmetries.push_back(sym);
	}
}


void TIRBM::setParameters(ParamSet set)
{
	this->parameters = set;
}


void TIRBM::train(vector<vector<double>>& input, int sample_size, int epoch)
{
	for (int ep = 0; ep < epoch; ep++) {
		contrastive_divergence(input, 1, sample_size);
	}
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
	std::ofstream weights;

	weights.open(filename + ".csv");
	for (int i = 0; i < n_vis; i++) {
		for (int j = 0; j < n_hid; j++) {
			if (j == n_hid - 1) {
				weights << this->wij[i][j];
				continue;
			}
			weights << this->wij[i][j] << ",";

		}
		weights << std::endl;
	}
	weights.flush();
	weights.close();
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
	if (p <= 0) return 0;
	if (p >= 1) return 1;

	int c = 0;
	double r;
	r = distribution(generator);

	if (r < p) c++;
	return c;
}

double TIRBM::uniform(double min, double max)
{
	return distribution(generator) * (max - min) + min;
}

void TIRBM::sample_h_given_v(vector<double>& vis_src, vector<vector<double>>& hid_target, vector<double>& hid_target_sample, vector<int>& max_pooled_s)
{
	
	vector<double> symPart(n_sym, 0);
	
	int num_hid = (int)n_hid;
	int num_vis = (int)n_vis;

	for (int i = 0; i < num_hid; i++) {
		double constPart = 0;
		for (int s = 0; s < n_sym; s++) {
			
			for (int j = 0; j < num_vis; j++) {
				//calculate the constant part only once
				
				constPart += this->wij[j][i] * ((*symmetries[s])(vis_src))[j];
				
				symPart[s] += this->wij[j][i] * ((*symmetries[s])(vis_src))[j];
			}
			symPart[s] += bjs[i][s];
			//again only compute the constant part once
			
			constPart += bjs[i][s];
			
			
		}
		for (int s = 0; s < n_sym; s++) {
			hid_target[i][s] = std::exp(symPart[s]) / (1 + std::exp(constPart));
		}
		max_pooled_s[i] = max_pool(hid_target[i]);
		hid_target_sample[i] = bernoulli(hid_target[i][max_pooled_s[i]]);
	}
}
vector<double> TIRBM::getColumn(int column) {
	vector<double> output;
	for (int i = 0; i < n_vis; i++) {
		output.push_back(wij[i][column]);
	}
	return output;
}
void TIRBM::sample_v_given_h(vector<double>& hid_src, vector<double>& vis_target, vector<double>& vis_target_sample, vector<int> &max_pooled_s)
{
	double pre_sigmoid = 0;
	int num_vis = (int)n_vis;
	int num_hid = (int)n_hid;

	for (int i = 0; i < num_vis; i++) {
		pre_sigmoid = 0;
			for (int j = 0; j < num_hid; j++) {
				auto column = getColumn(j);
				pre_sigmoid += ((*symmetries[max_pooled_s[j]])(column))[i] * hid_src[j];
			}
			pre_sigmoid += ci[i];
		vis_target[i] = actFun(pre_sigmoid);
		vis_target_sample[i] = bernoulli(vis_target[i]);
	}

}

double TIRBM::contrastive_divergence(vector<vector<double>>& input, int cdK, int batchSize)
{
	vector<double> tmpVisUpdate(n_vis);
	vector<vector<double>> tmpHidUpdate(n_hid, vector<double>(n_sym));
#pragma parallel for
	for (int numBatch = 0; numBatch < batchSize; numBatch++) {
		vector<vector<double>> hid0(n_hid, vector<double>(n_sym));
		vector<double> hid0_sampled(n_hid);
		vector<vector<double>> hidN(n_hid, vector<double>(n_sym));
		vector<double> hidN_sampled(n_hid);
		vector<double> visN(n_vis);
		vector<double> visN_sampled(n_vis);
		vector<int> max_pooled_s(n_vis);
		vector<int> max_pooled_sN(n_vis);
		//do CDk
		for (int i = 0; i < cdK; i++) {
			if (i == 0) {
				sample_h_given_v(input[numBatch], hid0, hid0_sampled, max_pooled_s);
				sample_v_given_h(hid0_sampled, visN, visN_sampled, max_pooled_sN);
			}
			else {
				sample_h_given_v(visN_sampled, hidN, hidN_sampled, max_pooled_sN);
				sample_v_given_h(hidN_sampled, visN, visN_sampled, max_pooled_sN);
			}
		}
		sample_h_given_v(visN_sampled, hidN, hidN_sampled, max_pooled_sN);
		//calculate the gradients for each batch
		for (int i = 0; i < n_vis; i++) {
			for (int j = 0; j < n_hid; j++) {
				double tmpW = this->wij[i][j];
				//update new delta
				double delta = 0;

				delta = this->parameters.lr * ((*symmetries[max_pooled_s[j]])(input[numBatch])[i] * hid0_sampled[j] - (*symmetries[max_pooled_sN[j]])(visN)[i] * hidN_sampled[j]);
				
				this->tmpDwij[i][j] += delta;

				//check for regularizer
				if (this->parameters.regulization & Regularization::L1) {
					//apply L1 regulizer
					int sign = std::signbit(tmpW) ? -1 : 1;
					this->tmpDwij[i][j] -= this->parameters.lr *this->parameters.weightDecay *sign;
				}
				if (this->parameters.regulization & Regularization::L2) {
					this->tmpDwij[i][j] -= this->parameters.lr *this->parameters.weightDecay * tmpW;
				}
			}
		}

		//update biases only use bias if not dropconnect
			for (int i = 0; i < n_vis; i++) {
				tmpVisUpdate[i] += this->parameters.lr * (input[numBatch][i] - visN_sampled[i]);
			}

			for (int j = 0; j < n_hid; j++) {
				tmpHidUpdate[j][max_pooled_sN[j]] += this->parameters.lr * (hid0_sampled[j] - hidN_sampled[j]);
			}
	}
	//apply the changes to the values
#pragma omp parallel for
	for (int i = 0; i < n_vis; i++) {
		for (int j = 0; j < n_hid; j++) {
			double tmpW = this->wij[i][j];
			this->tmpDwij[i][j] /= batchSize;
			//let the change flow
			//if average is activated, average the weights
			double newWeight = this->wij[i][j] + dWij[i][j] * this->parameters.momentum + tmpDwij[i][j];
			this->wij[i][j] = newWeight;
			//update new delta
			//apply current change
			//normalize with respect to batchsize, to flatten response

			dWij[i][j] = tmpDwij[i][j];
		}
	}
#pragma omp parallel for
	for (int i = 0; i < n_vis; i++) {
		ci[i] += tmpVisUpdate[i] / batchSize;
	}
#pragma omp parallel for
	for (int j = 0; j < n_hid; j++) {
		for (int s = 0; s < n_sym; s++) {
			this->bjs[j][s] += tmpHidUpdate[j][s] / batchSize;
		}
	}

	return 0.0;
}

int TIRBM::max_pool(vector<double> hid_fixedj)
{
	return std::distance(hid_fixedj.begin(),std::max_element(hid_fixedj.begin(), hid_fixedj.end()));
}



TIRBM::TIRBM(int n_vis, int n_hid, FunctionType activationFunction) :
	generator(time(NULL)),
	distribution(0.0,1.0), actFun(activationFunction),
	wij(n_vis,vector<double>(n_hid,uniform(-1,1))),
	dWij(n_vis, vector<double>(n_hid)),
	tmpDwij(n_vis, vector<double>(n_hid)),
	bjs(n_hid,vector<double>()),
	dbjs(n_hid, vector<double>()),
	ci(n_vis),
	dci(n_vis),
	n_hid(n_hid),
	n_vis(n_vis),
	n_sym(0)
{
	

}
