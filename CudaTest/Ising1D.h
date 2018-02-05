#pragma once
#include "LatticeObject.h"
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
public:
	Ising1D(int size);
	Ising1D(int size, double  beta, double J);
	~Ising1D();
	void monteCarloStep();
	vector<int> getConfiguration();
	double getMagnetization();
	double getMeanEnergy();
	double getTheoreticalMeanEnergy();
	unsigned int seed;
};

