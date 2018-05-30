#include "Phi4Test.h"
#include "Phi1D.h"
#include <iostream>
#include <fstream>
using namespace std;
Phi4Test::Phi4Test()
{
}


Phi4Test::~Phi4Test()
{
	
}

void Phi4Test::run()
{
	ofstream sus("susceptability.csv");
	vector<Phi1D*> phi4s;
	int steps = 1;
	for (int i = 0; i < steps; i++) {
		
		Phi1D *phi4 = new Phi1D(128 * pow(2,i), 1.0, 10000000000000, -1, 1);
		phi4->useWolff = true;

		phi4s.push_back(phi4);
	}
	
	for (double beta = 0; beta < 1000000000; beta += 100)
	{	
		vector<double> averages(steps);
#pragma omp parallel for
		for (int s = 0; s < steps; s++) {
			phi4s[s]->changeKappa(beta);
			double susce = 0.0;
			double phiSquared = 0.0;
			double avgAbsPhi = 0.0;
			for (int i = 0; i < 1000; i++) {
				auto tmp = phi4s[s]->volumeAverage();
				susce += abs(tmp);
				avgAbsPhi += phi4s[s]->absoluteVolumeAverage();
				phi4s[s]->monteCarloSweep();
			}
			
			//double susce = ( phiSquared - avgAbsPhi *avgAbsPhi);
			averages[s] = avgAbsPhi/1000.0;
			avgAbsPhi /= 100.0;
			std::cout << "\r" << "kappa: "<< beta <<  " avg abs phi: " << susce / 1000.0 << "                            ";
		}
		sus << beta << ",";
		for (int s = 0; s < steps; s++) {
			sus << averages[s] << ",";
		}
		sus << std::endl;
		
	}
	sus.close();


}

void Phi4Test::runCorrTest()
{
	for (int m = 0; m < 1000; m++) 
	{
		double kappa = 0.3;
		Phi1D phi(512, kappa, 0, 0, 0);
		double difference = 0;
		double quotient = 0;
		int trials = 1000;
		double corrLength = 0;
		phi.useWolff = true;
		//phi.thermalize();
		//phi.thermalize();
		double slope = 0;

		for (int i = 0; i < trials; i++) {
			phi.fftUpdate();
			slope += log(phi.getCorrelationLength(1)) - log(abs(phi.getCorrelationLength(2)));
			corrLength += phi.getCorrelationLength(1);
		}
		double m = sqrt(1.0 / kappa - 2);
		double scale = (corrLength / trials) / exp(-m);
		difference = corrLength / trials - exp(-m);
		std::cout << "Results" << std::endl;
	std:cout << "Quotient requirement: " << quotient / trials << std::endl;
		std::cout << "Difference requirement: " << difference << std::endl;
		std::cout << "Correlation length: " << corrLength / trials << " | " << exp(-m) << " -- " << scale << std::endl;
		std::cout << "m: " << m << "  slope: " << slope / trials << std::endl;
		std::cout << "Coefficient: " << log(pow(1.0 / (corrLength / trials), 1.0 / m)) << std::endl;
		std::cout << (abs(difference) < 1.0 / sqrt(trials) && abs(quotient / trials) < 1.0 / sqrt(trials) ? "PASSED" : "FAILED") << std::endl;

		std::cout << std::endl << std::endl;

		corrLength = 0;
		slope = 0;

		for (int i = 0; i < trials; i++) {
			phi.fftUpdate();
			phi.rescaleFields(sqrt(1.0 / scale));
			slope += log(phi.getCorrelationLength(1)) - log(abs(phi.getCorrelationLength(2)));
			corrLength += phi.getCorrelationLength(1);
		}
		m = sqrt(1.0 / kappa - 2);
		scale = (corrLength / trials) / exp(-m);
		difference = corrLength / trials - exp(-m);
		std::cout << "Results" << std::endl;
		std::cout << "Quotient requirement: " << quotient / trials << std::endl;
		std::cout << "Difference requirement: " << difference << std::endl;
		std::cout << "Correlation length: " << corrLength / trials << " | " << exp(-m) << " -- " << scale << std::endl;
		std::cout << "m: " << m << "  slope: " << slope / trials << std::endl;
		std::cout << "Coefficient: " << log(pow(1.0 / (corrLength / trials), 1.0 / m)) << std::endl;
		std::cout << (abs(difference ) < 1.0 / sqrt(trials) && abs(quotient / trials) < 1.0 / sqrt(trials) ? "PASSED" : "FAILED") << std::endl;
		std::cout << "----" << std::endl << std::endl;
	}
}
