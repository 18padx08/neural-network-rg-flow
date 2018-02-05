#include "RBM.h"

#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0, 1.0);

double sample(double p) {
	if (p < 0) return 0;
	if (p > 1) return 1;

	int c = 0;
	double r;
	r = distribution(generator);
	
	if (r < p) c++;
	return c;
}

double uniform(double min, double max) {
	return distribution(generator) * (max - min) + min;
}

RBM::RBM(int n_vis, int n_hid, FunctionType activationFunction) : actFun(activationFunction)

{
	this->n_vis = n_vis;
	this->n_hid = n_hid;
	vis_b = (double *)malloc(n_vis * sizeof(double));
	hid_b = (double *)malloc(n_hid * sizeof(double));


	this->W = (double **)malloc(n_vis * sizeof(double *));
	this->dropConnectMask = (bool **)malloc(n_vis * sizeof(bool *));
	this->dW = (double **)malloc(n_vis * sizeof(double *));
	for (int i = 0; i < n_vis; i++) {
		this->W[i] = (double*)malloc(sizeof(double) * n_hid);
		this->dropConnectMask[i] = (bool*)malloc(sizeof(bool) * n_hid);
		this->dW[i] = (double*)malloc(sizeof(double) * n_hid);
	}
	
	//srand(time(NULL));
	initWeights();
}

RBM::RBM(int n_vis, int n_hid) : RBM(n_vis, n_hid, FunctionType::SIGMOID)
{
}

void RBM::sample_h_given_v(double * vis_src, double * hid_target, double *hid_sampled)
{
	double pre_sigmoid = 0;
	//we would implement dropout or dropconnect here
	int num_hid = (int)n_hid;
	int num_vis = (int)n_vis;

	for (int i = 0; i < num_hid; i++) {
		for (int j = 0; j < num_vis; j++) {
			if(!(this->reg & Regulization::DROPCONNECT) || ((this->reg & Regulization::DROPCONNECT) && !this->dropConnectMask[j][i]))
				pre_sigmoid += this->W[j][i] * vis_src[j];
		}
		pre_sigmoid += hid_b[i];
		hid_target[i] = actFun(pre_sigmoid);
		hid_sampled[i] = sample(hid_target[i]);
	}
}

void RBM::sample_v_given_h(double * hid_src, double * vis_target, double * vis_sampled)
{
	double pre_sigmoid = 0;
	//we would implement dropout or dropconnect here
	int num_vis = (int)n_vis;
	int num_hid = (int)n_hid;

	for (int i = 0; i < num_vis; i++) {
		for (int j = 0; j < num_hid; j++) {
			if (!(this->reg & Regulization::DROPCONNECT) || ((this->reg & Regulization::DROPCONNECT) && !this->dropConnectMask[j][i]))
			pre_sigmoid += this->W[i][j] * hid_src[j];
		}
		pre_sigmoid += hid_b[i];
		vis_target[i] = actFun(pre_sigmoid);
		vis_sampled[i] = sample(vis_target[i]);
	}
}

double RBM::contrastive_divergence(double * input, int cdK, int batchSize)
{
	double *vis0_sampled = input;
	double *hid0_sampled = (double*)malloc(sizeof(double)*this->n_hid);
	double *hid0 = (double*)malloc(sizeof(double)*this->n_hid);
	double *visN = (double*)malloc(sizeof(double)*this->n_vis);
	double *hidN = (double*)malloc(sizeof(double)*this->n_hid);
	double *visN_sampled = (double*)malloc(sizeof(double)*this->n_vis);
	double *hidN_sampled = (double*)malloc(sizeof(double)*this->n_hid);

	//prepare first set
	sample_h_given_v(vis0_sampled, hid0, hid0_sampled);

	//calculate the "model" distribution
	for (int k = 0; k < cdK; k++) {
		sample_v_given_h(hid0_sampled, visN, visN_sampled);
		sample_h_given_v(visN_sampled, hidN, hidN_sampled);
	}
	int num_vis = (int)n_vis;
	int num_hid = (int)n_hid;
	//update weights

	for (int i = 0; i < num_vis; i++) {
		for (int j = 0; j < num_hid; j++) {
			double tmpW = this->W[i][j];
			//let the change flow
			this->W[i][j] += dW[i][j]*this->momentum;
			//update new delta
			this->dW[i][j] = this->lr * (vis0_sampled[i] * hid0[j] - visN_sampled[i] * hidN[j]) ;
			//check for regularizer
			if (this->reg & Regulization::L1) {
				//apply L1 regulizer
				int sign = std::signbit(tmpW) ? -1 : 1;
				this->dW[i][j] -= this->lr  *0.001* tmpW *sign;
			}
			//apply current change
			this->W[i][j] += dW[i][j]/batchSize;
			
		}
	}

	//update biases

	for (int i = 0; i < num_vis; i++) {
		this->vis_b[i] += this->lr * (vis0_sampled[i] - visN[i]);
	}

	for (int j = 0; j < num_hid; j++) {
		this->hid_b[j] += this->lr * (hid0_sampled[j] - hidN[j]);
	}

	auto ce = std::abs(crossEntropy(input) - crossEntropy(visN_sampled));

	
	delete(hid0_sampled);
	delete(hid0);
	delete(visN);
	delete(hidN);
	delete(visN_sampled);
	delete(hidN_sampled);
	return ce;
}

void RBM::initMask(bool **mask )
{
	if (mask == nullptr) {
		this->isRandom = true;
	}
	else {
		this->isRandom = false;
	}
	if (isRandom) {
		for (int i = 0; i < n_vis; i++) {
			for (int j = 0; j < n_hid; j++) {
				bool what = (bool)std::round(distribution(generator));
				this->dropConnectMask[i][j] = what;
			}
		}
	}
	else {
		for (int i = 0; i < n_vis; i++) {
			for (int j = 0; j < n_hid; j++) {
				this->dropConnectMask[i][j] = mask[i][j];
			}
		}
	}
}


//set biases to zero and apply uniform distribution to weights
void RBM::initWeights()
{
	for (int i = 0; i < n_vis; i++) {
		for (int j = 0; j < n_hid; j++) {
			this->W[i][j] = uniform(-1,1);
			this->dW[i][j] = 0;
			if (i == 0) {
				this->hid_b[j] = 0;
			}
		}
		this->vis_b[i] = 0;
	}
}
std::string printProg(int current, int max) {
	std::string progress = "";
	for (int i = 0; i < current; i++) {
		if (i % (int)std::max(1.0,std::round((max/100*5))) == 0) {
			progress += "=";
		}
	}
	return progress;
}
double RBM::crossEntropy(double *input) {
	double biasVis = 0;
	for (int i = 0; i < n_vis; i++) {
		biasVis += input[i] * vis_b[i];
	}
	double hid = 0;
	for (int j = 0; j < n_hid; j++) {
		double inner = hid_b[j];
		for (int i = 0; i < n_vis; i++) {
			inner += input[i] * this->W[i][j];
		}
		hid += std::log(1 + std::exp(inner));
	}
	return -biasVis - hid;
}
void RBM::train(double ** input, int sample_size, int epoch)
{
	double ce = 0;
	double average = 0;
	average = 0;
	int counter = 1;
	for (int ep = 0; ep < epoch; ep++) {
		//ereas line
		std::cout <<"\r" << "                                                                                                   ";
		//only do this if we are random init
		if(this->isRandom)
			this->initMask();
		average = 0;
		counter = 1;
		for (int i = 0; i < sample_size; i++)
		{
			average += contrastive_divergence(input[i], 1, sample_size) / sample_size;
			if(i%10 == 0)
				ce = average / counter;
			std::cout << "\r" << "epoch [" << ep + 1 << " (" << std::round((float)ep / epoch * 100) << "%)] " << "CrossEntropy: " << ce << " " << std::round((float)i / sample_size * 100) << "% [" << printProg(i, sample_size);
			counter++;
			//if isRandom we need to update the dropconnect mask
			if(this->isRandom) 
				this->initMask();
		}
		std::cout << "\r" << "epoch [" << ep+1 << "] " << "CrossEntropy: " << ce << " " <<  100 << "% [" << printProg(sample_size, sample_size) << "]";
	}
}

double * RBM::sample_from_net()
{
	double *visN = (double*)malloc(sizeof(double)*this->n_vis);
	double *hidN = (double*)malloc(sizeof(double)*this->n_hid);
	double *visN_sampled = new double[(sizeof(double)*this->n_vis)];
	double *hidN_sampled = (double*)malloc(sizeof(double)*this->n_hid);

	for (int i = 0; i < n_hid; i++) {
		hidN[i] = uniform(0, 1);
		hidN_sampled[i] = sample(hidN[i]);
		std::cout << hidN_sampled[i];
	}
	hidN_sampled[0] = 1;
	std::cout << std::endl;
	//stop any regullizer occuring at the propagation level
	auto tmp = this->reg;
	this->reg = Regulization::NONE;
	sample_v_given_h(hidN_sampled, visN, visN_sampled);
	//recover state
	this->reg = tmp;
	delete(hidN);
	delete(visN);
	delete(hidN_sampled);

	return visN_sampled;
}

double * RBM::reconstruct(double * input)
{
	double *vis0_sampled = input;
	double *hid0_sampled = (double*)malloc(sizeof(double)*this->n_hid);
	double *hid0 = (double*)malloc(sizeof(double)*this->n_hid);
	double *visN = (double*)malloc(sizeof(double)*this->n_vis);
	double *visN_sampled = (double*)malloc(sizeof(double)*this->n_vis);
	
	//stop any regullizer occuring at the propagation level
	auto tmp = this->reg;
	this->reg = Regulization::NONE;
	//prepare first set
	sample_h_given_v(vis0_sampled, hid0, hid0_sampled);
	sample_v_given_h(hid0_sampled, visN, visN_sampled);
	this->reg = tmp;
	delete(vis0_sampled);
	delete(hid0_sampled);
	delete(hid0);
	delete(visN);

	return visN_sampled;
}

void RBM::setParameters(ParamSet set)
{
	this->lr = set.lr;
	this->momentum = set.momentum;
	this->reg = set.regulization;
}
void RBM::printWeights() {
	for (int i = 0; i < n_vis; i++) {
		for (int j = 0; j < n_hid; j++) {
			std::cout << W[i][j];
		}
		std::cout << std::endl;
	}
}
void RBM::saveToFile(std::string filename)
{
	std::ofstream weights;
		
		weights.open(filename);
		for (int i = 0; i < n_vis; i++) {
			for (int j = 0; j < n_hid; j++) {
				weights << this->W[i][j] << ",";
			}
			weights << std::endl;
		}
		weights.flush();
		weights.close();
}

void RBM::saveVisualization()
{
	std::cout << "Saving visualization: 0%";
	for (int n = 0; n < n_hid; n++) {
		//generate a filter for each hidden neuron
		std::ofstream vis;
		std::string filename = "visNeuron" + std::to_string(n) + ".csv";
		vis.open(filename);
		for (int i = 0; i < n_vis; i++) {
			if (i != 0 && i % 28 == 0) {
				vis << std::endl;
			}
			vis << this->W[i][n] << ",";
		}
		vis.close();
		std::cout << "\r" << "Saving visualization: " << (int)(n / n_hid * 100) << "%";
	}
	std::cout << std::endl;
}

bool RBM::loadWeights(std::string filename)
{
	std::ifstream weights;
	weights.open(filename);
	//readline for line
	std::string deli = ",";
	int i=0 , j = 0;
	while (!weights.eof()) {
		std::string theLine = "";
		std::getline(weights, theLine);
		//split the line
		size_t pos = 0;
		int j = 0;
		while ((pos = theLine.find(deli)) != std::string::npos) {
			std::string Wij = theLine.substr(0, pos);
			if (Wij.length() > 0) {
				double wij = std::stod(Wij);
				if (i < this->n_vis && j < this->n_hid) {
					this->W[i][j] = wij;
				}
				else {
					throw std::exception("Weight-Matrix does not match dimension of NN");
				}
			}
			else {
				std::cout << "last element?" <<std::endl;
			}
			theLine.erase(0, pos + deli.length());
			j++;
		}
		i++;
	}
	if (i < this->n_vis) {
		throw std::exception("Weight-Matrix does not match dimension of NN");
	}
}
