#define _CRT_RAND_S  
#include <stdlib.h>     /* srand, rand */
#include "Ising1D.h"
#include "LatticeObject.cpp"

using namespace std;

double Ising1D::energyDiff(int index)
{
	double before = -J *(this->lattice[{index}] * this->lattice[{index + 1}] + this->lattice[{index - 1}] * this->lattice[{index}]);
	double after = -J * ((-1) * this->lattice[{index}] * this->lattice[{index + 1}] - this->lattice[{index - 1}] * this->lattice[{index}]);
	return after - before;
}

Ising1D::Ising1D(int size) : Ising1D(size, 0.01,-1) {}

Ising1D::Ising1D(int size, double beta, double J) : lattice({size}),  beta(beta), J(J)
{
	this->seed = time(NULL) + (int)&lattice.lattice;
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		rand_s(&this->seed);
		int n = this->seed % 2;
		if (n == 0) {
			lattice[{i}] = -1;
		}
		else {
			lattice[{i}] = 1;
		}
	}
}

Ising1D::~Ising1D()
{
}

void Ising1D::monteCarloStep()
{
	//get random index 
	rand_s(&this->seed);
	int index = this->seed % (2*this->lattice.dimensions[0]);
	double diff = energyDiff(index);
	double prob = (double)rand() / RAND_MAX;
	if (prob < min(1.0, exp(-diff*beta))) {
		this->lattice[{index}] *= -1;
	}
}

vector<int> Ising1D::getConfiguration()
{
	vector<int> v;
	for (int i = 0; i < this->lattice.latticeSize; i++) {
		if (this->lattice[{i}] == -1) {
			v.push_back(0);
		}
		else {
			v.push_back(this->lattice[{i}]);
		}
		
	}
	return v;
}

double Ising1D::getMagnetization()
{
	double mean = 0;
	for (int i = 0; i < this->lattice.latticeSize; i++) {
		mean += this->lattice[{i}];
	}
	return mean / this->lattice.latticeSize;
}

double Ising1D::getMeanEnergy()
{
	double energy = 0;
	for (int i = 0; i < this->lattice.latticeSize; i++) {
		double addi = -J*(this->lattice[{i - 1}] * this->lattice[{i}]);
		energy += addi;
	}
	return energy/this->lattice.latticeSize;
}

double Ising1D::getTheoreticalMeanEnergy() {
	double result = 0;
	int N = this->lattice.latticeSize;

	result += this->J ;
	//results in nan for too large lattices for large absolute values of j and beta just take -j
	result *= (1.0/(pow(cosh(this->beta*this->J),N) + pow(sinh(this->beta*this->J), N)));
	result *= pow(cosh(this->beta*this->J), N - 1) *sinh(this->beta*this->J) + pow(sinh(this->beta*this->J), N - 1) *cosh(this->beta*this->J);

	if (isnan(result)) {
		result = this->J;
	}
	return -result;
}
