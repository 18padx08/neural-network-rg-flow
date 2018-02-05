#include <string>
#pragma once
enum Regulization {
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
	Regulization regulization;
};

class RBM {
private:
	Regulization reg = Regulization::L1;
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
	void initMask();
	double crossEntropy(double *input);
	bool isRandom = false;
public:
	RBM(int n_vis, int n_hid);
	void initWeights();
	void train(double **input, int sample_size, int epoch);
	double * sample_from_net();
	double * reconstruct(double * input);
	void setParameters(ParamSet set);
	void printWeights();
	void saveToFile(std::string filename);
	void saveVisualization();
	bool loadWeights(std::string filename);
};
