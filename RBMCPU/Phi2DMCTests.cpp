#include "Phi2DMCTests.h"



Phi2DMCTests::Phi2DMCTests()
{
}


Phi2DMCTests::~Phi2DMCTests()
{
}

void Phi2DMCTests::criticalLineTest(vector<int> chainsize, vector<double> kappas, vector<double> lambdas)
{
	ofstream output("phi2dmctest.csv");
	for (double l : lambdas) {
		for (double k : kappas) {
			Phi2D phi(chainsize, k, l);
			phi.useWolff = true;
			phi.thermalize();
			
			double absAvg = 0;
			for (int i = 0; i < 50; i++) {
				absAvg += abs(phi.volumeAverage());
				phi.monteCarloSweep();
			}
			absAvg /= 50;
			output << l << "," << k << "," << absAvg << std::endl;
			std::cout << "At l=" << l << " k=" << k << " with vev=" << absAvg << std::endl;
			
			if (absAvg > 0.1) break;
			
		}
	}
}

void Phi2DMCTests::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	vector<int> chainsize = this->getIntVector("chainsize", num_vars, list_vars);
	vector<double> kappas = this->getDoubleVector("kappa", num_vars, list_vars);
	vector<double> lambdas = this->getDoubleVector("lambda", num_vars, list_vars);
	if (name == "criticalLineTest") {
		criticalLineTest(chainsize);
	}
}
