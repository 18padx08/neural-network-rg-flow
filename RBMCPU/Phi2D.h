#pragma once
#include <vector>
#include "../CudaTest/LatticeObject.h"
#include <algorithm>
#include <random>

#include <time.h>       /* time */
using namespace std;
class Phi2D
{
private:
	LatticeObject<double> lattice;
	double beta;
	double J;
	double energyDiff(int x,int y);
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
	double pi = 3.14159;
public:
	Phi2D(vector<int> size, double kappa, double lambda);
	~Phi2D();
	bool useWolff = false;
	void monteCarloStep();
	void monteCarloSweep();
	vector<vector<double>> getConfiguration();
	double volumeAverage();
	double absoluteVolumeAverage();
	double squaredVolumeAverage();
	void thermalize();
	double getEnergy();
	double getMagnetization();
	void rescaleFields(double scaling);
	double getCorrelationLength(int distance = 1);
};

