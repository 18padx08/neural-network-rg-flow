#include "Phi2D.h"
#include "../CudaTest/LatticeObject.cpp"

double Phi2D::energyDiff(int index)
{
	return 0.0;
}

void Phi2D::buildCluster()
{
}

void Phi2D::flipCluster()
{
}

Phi2D::Phi2D(int size, double kappa, double lambda, double m, double beta)
{
}

Phi2D::~Phi2D()
{
}

void Phi2D::monteCarloStep()
{
}

void Phi2D::monteCarloSweep()
{
}

vector<double> Phi2D::getConfiguration()
{
	return vector<double>();
}

double Phi2D::volumeAverage()
{
	return 0.0;
}

double Phi2D::absoluteVolumeAverage()
{
	return 0.0;
}

double Phi2D::squaredVolumeAverage()
{
	return 0.0;
}

void Phi2D::thermalize()
{
}

void Phi2D::changeBeta(double newBeta)
{
}

void Phi2D::changeLambda(double newLambda)
{
}

void Phi2D::changeKappa(double newKappa)
{
}

double Phi2D::susceptability()
{
	return 0.0;
}

double Phi2D::getEnergy()
{
	return 0.0;
}

double Phi2D::getMagnetization()
{
	return 0.0;
}
