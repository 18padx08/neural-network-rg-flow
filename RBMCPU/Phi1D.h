#pragma once
#include "../CudaTest/LatticeObject.h"
#include <algorithm>
#include <random>

#include <time.h>       /* time */
#include <vector>
using namespace std;
class Phi1D
{
private:
	LatticeObject<double> lattice;
	double beta;
	double J;
	double energyDiff(int index);
	std::uniform_int_distribution<int> dist;
	std::default_random_engine generator;
	void buildCluster();
	void flipCluster();
	int tau = -1;
	vector<int> cluster;
	int metropolisSweeps = 10;
	double kappa;
	double lambda;
	double m;
public:
	Phi1D(int size, double kappa, double lambda, double m, double beta);
	~Phi1D();
	bool useWolff = false;
	void monteCarloStep();
	void monteCarloSweep();
	vector<double> getConfiguration();
	double volumeAverage();
	double absoluteVolumeAverage();
	double squaredVolumeAverage();
	double total();
	double getMax();
	void thermalize();
	void changeBeta(double newBeta);
	void changeLambda(double newLambda);
	void changeKappa(double newKappa);
	double susceptability();
	double getEnergy();
	double getMagnetization();
};

