#include "Phi1D.h"
#include <complex>
#include "FFT.cpp"
#include <assert.h>
#include "../CudaTest/LatticeObject.cpp"


double Phi1D::energyDiff(int index)
{
	double delta = 0;
	auto theValue = lattice[{index}];
	uniform_real_distribution<double> deltaDist(-1.0, 1.0);
	delta = deltaDist(generator);
	double deltaE = -2.0*kappa *(lattice[{index - 1}] + lattice[{index + 1}]) * (delta)+(pow(delta, 2) + 2 * theValue * delta) + lambda * pow(pow(theValue + delta,2) -1, 2) - lambda * pow(pow(theValue, 2) - 1, 2);
	std::uniform_real_distribution<double> pro(0, 1);
	double prob = pro(generator);
	if (prob < min(1.0, exp(-deltaE))) {
		lattice[{index}] = theValue + delta;
	}
	return deltaE;
}

void Phi1D::buildCluster()
{
	cluster.clear();
	
	//choose random spin
	int index = dist(generator);
	//parallel loop left and right
#pragma omp parallel sections
	{
#pragma omp section 
		{
			std::uniform_real_distribution<double> prob(0, 1);
			//go left
			for (int i = index - 1; i >= 0; i--) {
				double pro = prob(generator);
				if (signbit((double)lattice[{i}]) == signbit((double)lattice[{index}]))
				{
					if (pro < 1.0 - exp(-2 * kappa* lattice[{i}] * lattice[{i+1}])) {
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
			for (int i = index + 1; i < this->lattice.latticeSize; i++) {
				double pro = prob(generator);
				if (signbit((double)lattice[{i}]) == signbit((double)lattice[{index}]))
				{
					if (pro < 1.0 - exp(-2 * kappa* lattice[{i}] * lattice[{i-1}])) {
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
	}

}

void Phi1D::flipCluster()
{
#pragma omp parallel for
	for (int i = 0; i < cluster.size(); i++) {
		this->lattice[{cluster[i]}] *= -1.0;
	}
}


Phi1D::Phi1D(int size, double kappa, double lambda, double m, double beta) : dist_for_fft(size), beta(beta), lattice({ size }), kappa(kappa), lambda(lambda), m(m), dist(0, size - 1), generator(time(NULL))
{
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		uniform_real_distribution<double> deltaDist(-1.5, 1.5);
		lattice[{i}] = deltaDist(generator);
		auto sigma =  1.0 / 2.0* sqrt(lattice.dimensions[0])/ std::sqrt((1 - 2*kappa*std::cos(i * 2 * pi / size)));
		//std::cout << sigma << std::endl;
		dist_for_fft[i] = std::normal_distribution<double>(0, sigma);
 	}
	this->tau = lattice.latticeSize / 10.0;
}

Phi1D::~Phi1D()
{
}

void Phi1D::monteCarloStep()
{
	//get random index 
	int index = dist(generator);
	
	double diff = energyDiff(index);
}

void Phi1D::monteCarloSweep()
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

vector<double> Phi1D::getConfiguration()
{
	vector<double> v;
	for (int i = 0; i < this->lattice.latticeSize; i++) {
		v.push_back(this->lattice[{i}]);
	}
	return v;
}

//if lambda is equals to zero, we can use the fact that the action can be diagonalized
void Phi1D::fftUpdate()
{
	assert(lambda == 0);
	auto rand = std::uniform_real_distribution<double>(0, 2 * pi);
	vector<complex<double>> v_hat(lattice.dimensions[0]);
	for (int i = 1; i <= lattice.dimensions[0]/2; i++) {
		auto tmp = dist_for_fft[i](generator);
		auto tmp2 = dist_for_fft[i](generator);
		if (i == lattice.dimensions[0] / 2) {
			v_hat[i] = { tmp, 0 };
			v_hat[lattice.dimensions[0] - i] = { tmp,0 };
		}
		else {
			v_hat[i] = { tmp, tmp2 };
			v_hat[lattice.dimensions[0] - i] = { tmp,-tmp2 };
		}
		//std::cout << v_hat[i] << std::endl;
	}
	v_hat[0] = 0;
	//fourier transform result
	auto result = fft::ifft(v_hat);
	vector<double> newRes(lattice.dimensions[0]);
	auto N = lattice.dimensions[0];
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		//result should be real
		//std::cout << result[i] << std::endl;
		result[i] *=1.0 / (double)N;
		if (abs(result[i].imag()) > 1e-10) {
			std::cout << "Error: imaginary part should be zero: " << result[i] << std::endl;
			//break;
		}
		
		lattice[{i}] =  result[i].real();
	}

}

double Phi1D::volumeAverage()
{
	auto average = 0.0;
#pragma omp parallel for reduction(+:average)
	for (int i = 0; i < lattice.latticeSize; i++) {
		average += lattice[{i}];
	}
	return average/lattice.latticeSize;
}

double Phi1D::absoluteVolumeAverage()
{
	auto average = 0.0;
#pragma omp parallel for reduction(+:average)
	for (int i = 0; i < lattice.latticeSize; i++) {
		average += abs(lattice[{i}]);
	}
	return average / lattice.latticeSize;
}

double Phi1D::squaredVolumeAverage()
{
	auto average = 0.0;
#pragma omp parallel for reduction(+:average)
	for (int i = 0; i < lattice.latticeSize; i++) {
		average += pow(lattice[{i}],2);
	}
	return average / lattice.latticeSize;
}

double Phi1D::total()
{
	auto average = 0.0;
#pragma omp parallel for reduction(+:average)
	for (int i = 0; i < lattice.latticeSize; i++) {
		average += lattice[{i}];
	}
	return average;
}

double Phi1D::getMax()
{
	double max = 0;
	for (int i = 0; i < lattice.latticeSize; i++) {
		if (abs(lattice[{i}]) > max) {
			max = abs(lattice[{i}]);
		}
	}
	return max;
}

void Phi1D::thermalize()
{
	for (int i = 0; i < 1000; i++) {
		this->monteCarloSweep();
	}
}

void Phi1D::changeBeta(double newBeta)
{
	beta = newBeta;
	this->thermalize();
}

void Phi1D::changeLambda(double newLambda)
{
	lambda = newLambda;
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		uniform_real_distribution<double> deltaDist(-1.5, 1.5);
		lattice[{i}] = deltaDist(generator);
	}
	thermalize();
}

void Phi1D::changeKappa(double newKappa)
{
	kappa = newKappa;
	for (int i = 0; i < lattice.dimensions[0]; i++) {

		uniform_real_distribution<double> deltaDist(-1.5, 1.5);
		lattice[{i}] = deltaDist(generator);
	}
	thermalize();
}

double Phi1D::susceptability()
{
	return lattice.latticeSize * (squaredVolumeAverage() - pow(volumeAverage(),2));
}

double Phi1D::getEnergy()
{
	return 0.0;
}

double Phi1D::getMagnetization()
{
	double mean = 0;
#pragma omp parallel for reduction(+:mean)
	for (int i = 0; i < this->lattice.latticeSize; i++) {
		mean += this->lattice[{i}];
	}
	return mean / this->lattice.latticeSize;
}

void Phi1D::rescaleFields(double scaling)
{
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		lattice[{i}] = scaling * lattice[{i}];
	}
}

double Phi1D::getCorrelationLength(int distance)
{
	double corrLength = 0;
	for (int i = 0; i < lattice.dimensions[0]; i++) {
		corrLength += lattice[{i}] * lattice[{i + distance}];
	}
	return corrLength / (lattice.dimensions[0]);
}

