#include <string>
#include <cmath>
#pragma once
enum Regularization {
	NONE = 0,
	L1 = 1,
	L2 = 1 << 1,
	SPARSE = 1 << 2,
	DROPOUT = 1 << 3,
	DROPCONNECT = 1 << 4
};

struct ParamSet {
	double lr;
	double momentum;
	Regularization regulization;
};
enum FunctionType {
	SIGMOID,
	TANH,
	RELU,

};

struct ActivationFunction {

private:
	FunctionType functionType = FunctionType::SIGMOID;
	double sigmoid(double arg) {
		return 1.0 / (1 + std::exp(-arg));
	}
	double relu(double arg) {
		return arg < 0 ? 0 : arg;
	}
	double tanh(double arg) {
		return std::tanh(arg);
	}
public:
	ActivationFunction(FunctionType type) {
		this->functionType = type;
	};
	double operator()(double arg) {
		switch (functionType) {
		case FunctionType::SIGMOID:
			return sigmoid(arg);
			break;
		case FunctionType::RELU:
			return relu(arg);
			break;
		case FunctionType::TANH:
			return tanh(arg);
			break;
		default:
			return sigmoid(arg);
		}
	}

};

class RBM {
private:
	Regularization reg = Regularization::L1;
	double **W;
	bool **dropConnectMask;
	double **dW;
	double *vis_b;
	double *hid_b;
	double lr = 0.01;
	double n_hid;
	double n_vis;
	double momentum=0.3;
	void sample_h_given_v(double *vis_src, double *hid_target, double *hid_target_sample);
	void sample_v_given_h(double *hid_src, double *vis_target, double *vis_target_sample);
	double contrastive_divergence(double *input, int cdK, int batchSize);
	ActivationFunction actFun;
	double crossEntropy(double *input);
	bool isRandom = true;
public:
	RBM(int n_vis, int n_hid);
	RBM(int n_vis, int n_hid, FunctionType activationFunction);
	void initWeights();
	void initMask(bool **mask = 0);
	void train(double **input, int sample_size, int epoch);
	double * sample_from_net();
	double * reconstruct(double * input);
	void setParameters(ParamSet set);
	void printWeights();
	void saveToFile(std::string filename);
	void saveVisualization();
	bool loadWeights(std::string filename);

};
