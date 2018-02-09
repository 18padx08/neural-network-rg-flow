#include "RBM.h"

#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std::chrono;
std::default_random_engine generator(time(NULL));
std::uniform_real_distribution<double> distribution(0.0, 1.0);

double sample(double p) {
	if (p <= 0) return 0;
	if (p >= 1) return 1;

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
	this->tmpdW = (double **)malloc(n_vis * sizeof(double *));
	for (int i = 0; i < n_vis; i++) {
		this->W[i] = (double*)malloc(sizeof(double) * n_hid);
		this->dropConnectMask[i] = (bool*)malloc(sizeof(bool) * n_hid);
		this->dW[i] = (double*)malloc(sizeof(double) * n_hid);
		this->tmpdW[i] = (double*)malloc(sizeof(double) * n_hid);
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
	int num_hid = (int)n_hid;
	int num_vis = (int)n_vis;

	for (int i = 0; i < num_hid; i++) {
		pre_sigmoid = 0;
		for (int j = 0; j < num_vis; j++) {
			if(!(this->reg & Regularization::DROPCONNECT) || ((this->reg & Regularization::DROPCONNECT) && this->dropConnectMask[j][i]))
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
		pre_sigmoid = 0;
		for (int j = 0; j < num_hid; j++) {
			if (!(this->reg & Regularization::DROPCONNECT) || ((this->reg & Regularization::DROPCONNECT) && this->dropConnectMask[i][j]))
			pre_sigmoid += this->W[i][j] * hid_src[j];
		}
		pre_sigmoid += hid_b[i];
		vis_target[i] = actFun(pre_sigmoid);
		vis_sampled[i] = sample(vis_target[i]);
	}
}

double RBM::contrastive_divergence(double ** input, int cdK, int batchSize)
{
	double *tmpHidUpdate = (double*)malloc(sizeof(double)*this->n_hid);
	double *tmpVisUpdate = (double*)malloc(sizeof(double)*this->n_vis);
	
	double ce = 0;
	//init 
	int num_vis = (int)n_vis;
	int num_hid = (int)n_hid;

#pragma omp parallel for
	for (int sampleNum = 0; sampleNum < batchSize; sampleNum++) {
		double *vis0_sampled = input[sampleNum];
		//allocate memory
		double *hid0_sampled = (double*)malloc(sizeof(double)*num_hid);
		double *hid0 = (double*)malloc(sizeof(double)*num_hid);
		double *visN = (double*)malloc(sizeof(double)*num_vis);
		double *hidN = (double*)malloc(sizeof(double)*num_hid);
		double *visN_sampled = (double*)malloc(sizeof(double)*num_vis);
		double *hidN_sampled = (double*)malloc(sizeof(double)*num_hid);
		

		for (int i = 0; i < num_vis; i++) {
			tmpVisUpdate[i] = 0;
		}

		for (int i = 0; i < num_hid; i++) {
			tmpHidUpdate[i] = 0;
		}

		//prepare first set
		sample_h_given_v(vis0_sampled, hid0, hid0_sampled);

		//calculate the "model" distribution
		for (int k = 0; k < cdK; k++) {
			if (k == 0) {
				sample_v_given_h(hid0_sampled, visN, visN_sampled);
			}
			else {
				sample_v_given_h(hidN_sampled, visN, visN_sampled);
			}
			sample_h_given_v(visN_sampled, hidN, hidN_sampled);
		}
		
		//update weights
		//only update if we dont do any dropconnect
		//accumulate the gradients

		for (int i = 0; i < num_vis; i++) {
			for (int j = 0; j < num_hid; j++) {
				if (!(this->reg & Regularization::DROPCONNECT) || ((this->reg & Regularization::DROPCONNECT) && this->dropConnectMask[i][j])) {
					double tmpW = this->W[i][j];
					bool tmp1 = !(this->reg & Regularization::DROPCONNECT);
					bool tmp2 = ((this->reg & Regularization::DROPCONNECT) && this->dropConnectMask[i][j]);
					bool tmp3 = tmp1 || tmp2;
					//update new delta
					double delta = 0;
					delta = this->lr * (vis0_sampled[i] * hid0[j] - visN[i] * hidN[j]);

					this->tmpdW[i][j] += delta;

					//check for regularizer
					if (this->reg & Regularization::L1) {
						//apply L1 regulizer
						int sign = std::signbit(tmpW) ? -1 : 1;
						this->tmpdW[i][j] -= this->lr*this->weight_decay *sign;
					}
					if (this->reg & Regularization::L2) {
						this->tmpdW[i][j] -= this->lr*this->weight_decay * tmpW;
					}
				}
			}
		}

		//update biases only use bias if not dropconnect
		if (!(Regularization::DROPCONNECT & this->reg)) {
			for (int i = 0; i < num_vis; i++) {
				tmpVisUpdate[i] += this->lr * (vis0_sampled[i] - visN[i]);
			}

			for (int j = 0; j < num_hid; j++) {
				tmpHidUpdate[j] += this->lr * (hid0_sampled[j] - hidN[j]);
			}
		}
		//calculate cross entropy
		double before = crossEntropy(vis0_sampled);
		double after = crossEntropy(visN_sampled);
		ce += std::abs( after - before );

		delete(hid0_sampled);
		delete(hid0);
		delete(visN);
		delete(hidN);
		delete(visN_sampled);
		delete(hidN_sampled);
	}

	//apply the average of the gradients

#pragma omp parallel for
	for (int i = 0; i < num_vis; i++) {
		for (int j = 0; j < num_hid; j++) {
			if (!(this->reg & Regularization::DROPCONNECT) || (this->reg & Regularization::DROPCONNECT) && this->dropConnectMask[i][j]) {
				double tmpW = this->W[i][j];
				this->tmpdW[i][j] /= batchSize;
				//let the change flow
				this->W[i][j] += dW[i][j] * this->momentum;
				//update new delta
				//apply current change
				//normalize with respect to batchsize, to flatten response
				this->W[i][j] += tmpdW[i][j];
				dW[i][j] = tmpdW[i][j];
				
			}
		}
	}

	//rescale the weights
	
	double *length = (double *) malloc(sizeof(double)*num_hid);
#pragma omp parallel for
	for (int i = 0; i < num_hid; i++) {
		for (int j = 0; j < num_vis; j++) {
			length[i] += std::pow(this->W[j][i],2);
		}
		length[i] = std::sqrt(length[i]);
		if (length[i] > n_hid) {
			for (int j = 0; j < num_vis; j++) {
				this->W[j][i] /= length[i];
			}
		}
	}
	
	//update biases only use bias if not dropconnect
	if (!(Regularization::DROPCONNECT & this->reg)) {
		for (int i = 0; i < n_vis; i++) {
			this->vis_b[i] += tmpVisUpdate[i] / batchSize;
		}

		for (int j = 0; j < n_hid; j++) {
			this->hid_b[j] += tmpHidUpdate[j]/batchSize;
		}
	}

	ce /= batchSize;
	//cleanup
	
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
				bool what = (bool)sample(distribution(generator));
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
			if (this->isRandom || (this->reg & Regularization::DROPCONNECT && this->dropConnectMask[i][j])) {
				this->W[i][j] =  uniform(-1, 1);
			}
			else {
				this->W[i][j] = 0;
			}
			this->dW[i][j] = 0;
			this->tmpdW[i][j] = 0;
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
	milliseconds loopStart  = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	for (int ep = 0; ep < epoch; ep++) {
		
		//ereas line
		//std::cout <<"\r" << "                                                                                                   ";
		//only do this if we are random init
		if(this->isRandom)
			this->initMask();
		//we need to average over all 
		average += contrastive_divergence(input, 1, sample_size);
		milliseconds now = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
		//std::cout << "\r" << "epoch [" << ep + 1 << " (" << std::round((float)ep / epoch * 100) << "%)] " << (now - loopStart).count()/60 << "s passed (" << ((now - loopStart) / (ep +1)).count() << "ms / loop)"  << "CrossEntropy: " << average / counter  << " [" << printProg(ep, epoch);
		counter++;
		//if isRandom we need to update the dropconnect mask
		if(this->isRandom) 
			this->initMask();
		//std::cout << "\r" << "epoch [" << ep+1 << "] " << "CrossEntropy: " << ce << " " <<  100 << "% [" << printProg(sample_size, sample_size) << "]";
	}
}

double * RBM::sample_from_net(int gibbs_steps)
{
	double *visN = (double*)malloc(sizeof(double)*this->n_vis);
	double *hidN = (double*)malloc(sizeof(double)*this->n_hid);
	double *visN_sampled = new double[this->n_vis];
	double *hidN_sampled = (double*)malloc(sizeof(double)*this->n_hid);

	for (int i = 0; i < n_hid; i++) {
		hidN[i] = uniform(0, 1);
		hidN_sampled[i] = sample(hidN[i]);
		std::cout << hidN_sampled[i];
	}
	//hidN_sampled[3] = 1;
	//hidN_sampled[5] = 1;
	std::cout << std::endl;
	//stop any regullizer occuring at the propagation level
	auto tmp = this->reg;
	this->reg = Regularization::NONE;
	for (int i = 0; i < gibbs_steps; i++) {
		sample_v_given_h(hidN_sampled, visN, visN_sampled);
		sample_h_given_v(visN, hidN, hidN_sampled);
	}
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
	double *visN = new double[this->n_vis];
	double *visN_sampled = (double*)malloc(sizeof(double)*this->n_vis);
	
	//stop any regullizer occuring at the propagation level
	auto tmp = this->reg;
	this->reg = Regularization::NONE;
	//prepare first set
	for (int i = 0; i < 10; i++) {
		sample_h_given_v(vis0_sampled, hid0, hid0_sampled);
		sample_v_given_h(hid0_sampled, visN, visN_sampled);
	}
	this->reg = tmp;
	//delete(vis0_sampled);
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
	this->weight_decay = set.weightDecay;
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
				if (j == n_hid - 1) {
					weights << this->W[i][j];
					continue;
				}
				weights << this->W[i][j] << ",";

			}
			weights << std::endl;
		}
		weights.flush();
		weights.close();

		std::ofstream hidden_bias;
		hidden_bias.open("hb" + filename);
		for (int i = 0; i < n_hid; i++) {
			if (i == n_hid - 1) {
				hidden_bias << this->hid_b[i];
				continue;
			}
			hidden_bias << this->hid_b[i] << ",";
		}
		hidden_bias.flush();
		hidden_bias.close();

		std::ofstream visible_bias;
		visible_bias.open("vb" + filename);
		for (int i = 0; i < n_vis; i++) {
			if (i == n_vis - 1) {
				visible_bias << this->hid_b[i];
				continue;
			}
			visible_bias << this->hid_b[i] << ",";
		}
		visible_bias.flush();
		visible_bias.close();

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
		double wij = 0;
		while ((pos = theLine.find(deli)) != std::string::npos) {
			std::string Wij = theLine.substr(0, pos);
			if (Wij.length() > 0) {
				wij = std::stod(Wij);
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
		if (theLine.length() > 0 && theLine != deli) {
			wij = std::stod(theLine);
			if (i < this->n_vis && j < this->n_hid) {
				this->W[i][j] = wij;
			}
			else {
				throw std::exception("Weight-Matrix does not match dimension of NN");
			}
		}
		i++;
	}
	if (i < this->n_vis) {
		throw std::exception("Weight-Matrix does not match dimension of NN");
	}
	//load biases
	std::ifstream visible_bias;
	visible_bias.open("vb" + filename);
	std::string theL = "";
	std::getline(visible_bias, theL);
	i = 0;
	size_t pos;
	while ((pos = theL.find(deli)) != std::string::npos) {
		std::string vb = theL.substr(0, pos);
		if (vb.length() > 0) {
			double wij = std::stod(vb);
			if (i < this->n_vis) {
				this->vis_b[i] = wij;
			}
			else {
				throw std::exception("Bias-Matrix does not match dimension of NN");
			}
		}
		else {
			std::cout << "last element?" << std::endl;
		}
		theL.erase(0, pos + deli.length());
		if (theL.length() > 0 && theL != deli) {
			double wij = std::stod(theL);
			if (i < this->n_vis) {
				this->vis_b[i] = wij;
			}
			else {
				throw std::exception("Bias-Matrix does not match dimension of NN");
			}
		}
		i++;
	}
	std::ifstream hidden_bias;
	hidden_bias.open("vb" + filename);
	std::getline(hidden_bias, theL);
	i = 0;
	while ((pos = theL.find(deli)) != std::string::npos) {
		std::string vb = theL.substr(0, pos);
		if (vb.length() > 0) {
			double wij = std::stod(vb);
			if (i < this->n_vis) {
				this->vis_b[i] = wij;
			}
			else {
				throw std::exception("Bias-Matrix does not match dimension of NN");
			}
		}
		else {
			std::cout << "last element?" << std::endl;
		}
		theL.erase(0, pos + deli.length());
		if (theL.length() > 0 && theL != deli) {
			double wij = std::stod(theL);
			if (i < this->n_vis) {
				this->vis_b[i] = wij;
			}
			else {
				throw std::exception("Bias-Matrix does not match dimension of NN");
			}
		}
		i++;
	}
}

double ** RBM::propagate(double ** input, int sample_size)
{
	double **output = (double **)malloc(sample_size * sizeof(double*));

	for (int i = 0; i < sample_size; i++) {
		output[i] = (double *)malloc(this->n_hid * sizeof(double));
		double *tmp = (double *)malloc(this->n_hid * sizeof(double));
		sample_h_given_v(input[i], tmp, output[i]);
		delete(tmp);
	}
	return output;
}

double * RBM::propup(double * hidden_activation, int gibbs_steps)
{
	double *output = (double *)malloc(sizeof(double)*n_vis);
	double *tmp = (double *)malloc(sizeof(double)*n_vis);
	double *tmp_hid = (double *)malloc(sizeof(double)*n_hid);
	double *tmp_hid_sampled = (double *)malloc(sizeof(double)*n_hid);
	
	sample_v_given_h(hidden_activation, tmp, output);
	for (int i = 1; i < gibbs_steps; i++) {
		sample_h_given_v(output, tmp_hid, tmp_hid_sampled);
		sample_v_given_h(tmp_hid_sampled, tmp, output);
	}
	delete(tmp);
	delete(tmp_hid);
	delete(tmp_hid_sampled);
	delete(hidden_activation);
	return output;
}
