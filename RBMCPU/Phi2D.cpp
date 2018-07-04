#include "Phi2D.h"

double Phi2D::energyDiff(int x, int y)
{
	double delta = 0;
	auto theValue = lattice[{x, y}];
	uniform_real_distribution<double> deltaDist(-1.0, 1.0);
	delta = deltaDist(generator);
	double deltaE = (pow(theValue, 2) - pow(theValue + delta, 2)) + lambda * (pow(pow(theValue, 2) - 1, 2) - pow(pow(theValue + delta, 2) - 1, 2));
	deltaE -= 2 * kappa * (lattice[{x + 1, y}] + lattice[{x - 1, y}] + lattice[{x + 1, y}] + lattice[{x - 1, y}]) * delta;
	std::uniform_real_distribution<double> pro(0, 1);
	double prob = pro(generator);
	if (prob < min(1.0, exp(-deltaE))) {
		lattice[{x,y}] = theValue + delta;
	}
	return deltaE;
}

void Phi2D::buildCluster()
{
	cluster.clear();

	//choose random spin
	uniform_int_distribution<int> xDirection(0, lattice.dimensions[0]);
	uniform_int_distribution<int> yDirection(0, lattice.dimensions[1]);
	int x = xDirection(generator);
	int y = yDirection(generator);
	//parallel loop left and right
#pragma omp parallel sections
	{
#pragma omp section 
		{
			std::uniform_real_distribution<double> prob(0, 1);

			//go left
			for (int i = x - 1; i >= 0; i--) {
				double pro = prob(generator);
				if (signbit((double)lattice[{i,y}]) == signbit((double)lattice[{x,y}]))
				{
					if (pro < 1.0 - exp(-2 * kappa* lattice[{i,y}] * lattice[{i + 1,y}])) {
						//add to cluster
#pragma omp critical
						cluster.push_back(i);
					}
					else {
						break;
					}
				}
				else {
					break;
				}
			}
		}
#pragma omp section
		{
			std::uniform_real_distribution<double> prob(0, 1);
			//go right
			for (int i = x + 1; i < this->lattice.dimensions[0]; i++) {
				double pro = prob(generator);
				if (signbit((double)lattice[{i,y}]) == signbit((double)lattice[{x,y}]))
				{
					if (pro < 1.0 - exp(-2 * kappa* lattice[{i,y}] * lattice[{i - 1,y}])) {
						//add to cluster
#pragma omp critical
						cluster.push_back(i);
					}
					else {
						break;
					}
				}
				else {
					break;
				}
			}
		}
#pragma omp section
		{
			std::uniform_real_distribution<double> prob(0, 1);
			//go up
			for (int j = y -1; j < 0; j++) {
				double pro = prob(generator);
				if (signbit((double)lattice[{x, j}]) == signbit((double)lattice[{x, y}]))
				{
					if (pro < 1.0 - exp(-2 * kappa* lattice[{x, j}] * lattice[{x, j+1}])) {
						//add to cluster
#pragma omp critical
						cluster.push_back(j);
					}
					else {
						break;
					}
				}
				else {
					break;
				}
			}
		}
#pragma omp section
		{
			std::uniform_real_distribution<double> prob(0, 1);
			//go down
			for (int j = y + 1; j < this->lattice.dimensions[1]; j++) {
				double pro = prob(generator);
				if (signbit((double)lattice[{x, j}]) == signbit((double)lattice[{x, y}]))
				{
					if (pro < 1.0 - exp(-2 * kappa* lattice[{x, j}] * lattice[{x, j-1}])) {
						//add to cluster
#pragma omp critical
						cluster.push_back(j);
					}
					else {
						break;
					}
				}
				else {
					break;
				}
			}
		}
	}
}

void Phi2D::flipCluster()
{
#pragma omp parallel for
	for (int i = 0; i < cluster.size(); i++) {
		this->lattice[{cluster[i]}] *= -1.0;
	}
}

Phi2D::Phi2D(vector<int> size, double kappa, double lambda) : kappa(kappa), lambda(lambda), lattice(size), generator(time(NULL)), dist(0,1)
{
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		for (int j = 0; j < lattice.dimensions[1]; j++) {
			uniform_real_distribution<double> deltaDist(-1.5, 1.5);
			lattice[{i,j}] = deltaDist(generator);
		}
	}
	this->tau = lattice.latticeSize / 10.0;
}

Phi2D::~Phi2D()
{
}

void Phi2D::monteCarloStep()
{
	//get random index 
	uniform_int_distribution<int> xDirection(0, lattice.dimensions[0]);
	uniform_int_distribution<int> yDirection(0, lattice.dimensions[1]);
	int x = xDirection(generator);
	int y = yDirection(generator);

	double diff = energyDiff(x,y);
}

void Phi2D::monteCarloSweep()
{
	//one monteCarlo sweep means going montecarlostep for 5 times lattice sides then wolff update
	for (int i = 0; i < metropolisSweeps * this->lattice.latticeSize; i++) {
		this->monteCarloStep();
	}
	//now build clusters
	if (useWolff) {
		buildCluster();
		//std::cout << "Size of cluster: " << cluster.size() <<std::endl;
		flipCluster();
	}
}

vector<vector<double>> Phi2D::getConfiguration()
{
	vector<vector<double>> v;
	for (int i = 0; i < this->lattice.dimensions[0]; i++) {
		vector<double> inner;
		for (int j = 0; j < this->lattice.dimensions[1]; j++) {
			inner.push_back(this->lattice[{i, j}]);
		}
		v.push_back(inner);
	}
	return v;
}

double Phi2D::volumeAverage()
{
	auto average = 0.0;
#pragma omp parallel for reduction(+:average)
	for (int i = 0; i < lattice.dimensions[0]; i++) {
#pragma omp parallel for reduction(+:average)
		for (int j = 0; j < lattice.dimensions[1]; j++) {
			average += lattice[{i,j}];
		}
	}
	return average / lattice.latticeSize;
}

double Phi2D::absoluteVolumeAverage()
{
	auto average = 0.0;
#pragma omp parallel for reduction(+:average)
	for (int i = 0; i < lattice.latticeSize; i++) {
		average += abs(lattice[{i}]);
	}
	return average / lattice.latticeSize;
}

double Phi2D::squaredVolumeAverage()
{
	auto average = 0.0;
#pragma omp parallel for reduction(+:average)
	for (int i = 0; i < lattice.latticeSize; i++) {
		average += pow(lattice[{i}], 2);
	}
	return average / lattice.latticeSize;
}

void Phi2D::thermalize()
{
	std::cout << std::endl;
	for (int i = 0; i < 100; i++) {
		this->monteCarloSweep();
		std::cout << "\r" << "                                         ";
		std::cout << "\r" << "[" << i << "]";
	}
	std::cout << std::endl;
}

double Phi2D::getEnergy()
{
	return 0.0;
}

double Phi2D::getMagnetization()
{
	return 0.0;
}

void Phi2D::rescaleFields(double scaling)
{
}

double Phi2D::getCorrelationLength(int distance)
{
	return 0.0;
}
