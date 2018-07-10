#include "Phi2DMCTests.h"



Phi2DMCTests::Phi2DMCTests()
{
}


Phi2DMCTests::~Phi2DMCTests()
{
}

void Phi2DMCTests::criticalLineTest(vector<int> chainsize, vector<double> kappas, vector<double> lambdas)
{
#pragma omp parallel for
	for (int i = 0; i < lambdas.size(); i++) {
		double l = lambdas[i];
		double k = kappas[0];
		int counter = 0;
		ofstream output("phi2dmctest_lambda=" + to_string(l) + ".csv");
		//take first value as base
		double baseValue = 0.1;
		bool first = true;
		while(true) {
			Phi2D phi(chainsize, k, l);
			phi.useWolff = true;
			phi.thermalize();
			
			double absAvg = 0;
			for (int i = 0; i < 100; i++) {
				absAvg += abs(phi.volumeAverage());
				phi.monteCarloSweep();
			}
			absAvg /= 100;
			if (first)
			{
				baseValue = absAvg*10;
				first = false;

			}
			output << l << "," << k << "," << absAvg << std::endl;
			std::cout << "At l=" << l << " k=" << k << " with vev=" << absAvg << std::endl;
			
			if (absAvg > baseValue) {
				counter++;
				if (counter > 5) break;
			}
			k += 0.02;
			
		}
	}
}

void Phi2DMCTests::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	vector<int> chainsize = this->getIntVector("chainsize", num_vars, list_vars);
	vector<double> kappas = this->getDoubleVector("kappa", num_vars, list_vars);
	vector<double> lambdas = this->getDoubleVector("lambda", num_vars, list_vars);
	if (name == "criticalLineTest") {
		criticalLineTest(chainsize, kappas, lambdas);
	}
}
