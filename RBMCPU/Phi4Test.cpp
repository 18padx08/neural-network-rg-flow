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
