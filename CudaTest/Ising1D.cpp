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


void Ising1D::buildCluster()
{
	cluster.clear();
	std::uniform_real_distribution<double> prob(0, 1);
	//choose random spin
	int index = dist(generator);
	//parallel loop left and right
#pragma omp parallel sections
	{
#pragma omp section 
		{
			//go left
			for (int i = index-1; i > 0; i--) {
				double pro = prob(generator);
				if (signbit((double)lattice[{i}]) == signbit((double)lattice[{index}]))
				{
					if (pro < 1.0 - exp(-2 * beta * J)) {
						//add to cluster
						cluster.push_back(i);
					}
				}
				else {
					break;
				}
			}
		}
#pragma omp section
		{
			//go right
			for (int i = index + 1; i < this->lattice.latticeSize; i++) {
				double pro = prob(generator);
				if (signbit((double)lattice[{i}]) == signbit((double)lattice[{index}])) 
				{
					if (pro < 1.0 - exp(-2 * beta * J)) {
						//add to cluster
						cluster.push_back(i);
					}
				}
				else {
					break;
				}
			}
		}
	}
}

void Ising1D::flipCluster()
{
#pragma omp parallel for
	for (int i = 0; i < cluster.size(); i++) {
		this->lattice[{cluster[i]}] *= -1;
	}
}

Ising1D::Ising1D(int size) : Ising1D(size, 0.01,-1) {}

Ising1D::Ising1D(int size, double beta, double J) : lattice({size}),  beta(beta), J(J), dist(0,size-1),generator(time(NULL))
{
	this->seed = time(NULL) + (int)&lattice.lattice;
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		seed = dist(generator);
		int n = this->seed % 2;
		if (n == 0) {
			lattice[{i}] = -1;
		}
		else {
			lattice[{i}] = 1;
		}
	}
	this->tau = lattice.latticeSize/10;
}

Ising1D::~Ising1D()
{
	//this->lattice.~LatticeObject();
}


void Ising1D::monteCarloStep()
{
	//get random index 
	this->seed = dist(generator);
	int index = this->seed;
	double diff = energyDiff(index);
	std::uniform_real_distribution<double> dist(0, 1);
	double prob = dist(generator);
	if (prob < min(1.0, exp(-diff*beta))) {
		this->lattice[{index}] *= -1;
	}
}

void Ising1D::monteCarloSweep()
{
	//one monteCarlo sweep means going montecarlostep for 5 times lattice sides then wolff update
	for (int i = 0; i < metropolisSweeps * this->lattice.latticeSize; i++) {
		this->monteCarloStep();
	}
	//now build clusters
	if (useWolff) {
		buildCluster();
		flipCluster();
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
#pragma omp parallel for reduction(+:mean)
	for (int i = 0; i < this->lattice.latticeSize; i++) {
		mean += this->lattice[{i}];
	}
	return mean / this->lattice.latticeSize;
}

double Ising1D::getMeanEnergy()
{
	int energy = 0;
#pragma omp parallel for reduction(+:energy)
	for (int i = 0; i < this->lattice.latticeSize; i++) {
		int addi = -(this->lattice[{i - 1}] * this->lattice[{i}]);
		energy += addi;
	}
	return J*(double)energy/this->lattice.latticeSize;
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

double Ising1D::calcExpectationValue(int n)
{
//	auto tau = calcAutoCorrelationTime();
	int value = 0;
	for (int i = 0; i < 10*this->lattice.latticeSize; i++) {
		//ensure that we have no correlation
		for (int j = 0; j < 5*tau; j++) {
			this->monteCarloStep();
		}
		//measure observable
		auto randomIndex = this->dist(generator);
		value += this->lattice[{randomIndex}] * this->lattice[{randomIndex + n}];
	}
	return (double)value/(10*this->lattice.latticeSize);
}

double Ising1D::calcAutoCorrelationTime()
{
	//calculate autocorrelation time go to 5 times auto int

	return 0.0;
}
