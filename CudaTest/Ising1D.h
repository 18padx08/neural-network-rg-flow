#pragma once
#include "LatticeObject.h"
#include <algorithm>
#include <random>

#include <time.h>       /* time */
#include <vector>
using namespace std;
class Ising1D
{
private:
	LatticeObject<int> lattice;
	double beta;
	double J;
	double energyDiff(int index);
	std::uniform_int_distribution<int> dist;
	std::default_random_engine generator;
	int tau = -1;
public:
	Ising1D(int size);
	Ising1D(int size, double  beta, double J);
	~Ising1D();
	void monteCarloStep();
	vector<int> getConfiguration();
	double getMagnetization();
	double getMeanEnergy();
	double getTheoreticalMeanEnergy();
	double calcExpectationValue(int n = 1);
	double calcAutoCorrelationTime();
	unsigned int seed;
};

