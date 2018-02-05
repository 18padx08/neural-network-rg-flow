#pragma once
class RBM
{
public:
	RBM(int n_visible, int n_hidden, bool *mask);
	~RBM();
	void train(int **samples, int n_samples);
	void printWeights();
private:
	int n_visible;
	int n_hidden;
	bool *mask;
	double *weights;
	double *bh;
	double *bv;
	double mean;
};

